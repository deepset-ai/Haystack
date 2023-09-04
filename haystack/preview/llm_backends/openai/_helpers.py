from typing import List, Callable, Dict, Any, Tuple
import os
import logging
import json

import tenacity
import requests
import sseclient

from haystack.preview.lazy_imports import LazyImport
from haystack.preview.llm_backends.chat_message import ChatMessage
from haystack.preview.llm_backends.openai.errors import OpenAIError, OpenAIRateLimitError, OpenAIUnauthorizedError

with LazyImport("Run 'pip install tiktoken'") as tiktoken_import:
    import tiktoken


logger = logging.getLogger(__name__)


OPENAI_TIMEOUT = float(os.environ.get("HAYSTACK_REMOTE_API_TIMEOUT_SEC", 30))
OPENAI_BACKOFF = int(os.environ.get("HAYSTACK_REMOTE_API_BACKOFF_SEC", 10))
OPENAI_MAX_RETRIES = int(os.environ.get("HAYSTACK_REMOTE_API_MAX_RETRIES", 5))
OPENAI_TOKENIZERS = {
    **tiktoken.model.MODEL_TO_ENCODING,
    "gpt-35-turbo": "cl100k_base",  # https://github.com/openai/tiktoken/pull/72
}
OPENAI_TOKENIZERS_TOKEN_LIMITS = {
    "text-davinci": 4097,  # Ref: https://platform.openai.com/docs/models/gpt-3
    "gpt-35-turbo": 4097,  # Ref: https://platform.openai.com/docs/models/gpt-3-5
    "gpt-3.5-turbo": 4097,  # Ref: https://platform.openai.com/docs/models/gpt-3-5
    "gpt-3.5-turbo-16k": 16384,  # Ref: https://platform.openai.com/docs/models/gpt-3-5
    "gpt-3": 4096,  # Ref: https://platform.openai.com/docs/models/gpt-3
    "gpt-4-32k": 32768,  # Ref: https://platform.openai.com/docs/models/gpt-4
    "gpt-4": 8192,  # Ref: https://platform.openai.com/docs/models/gpt-4
}
OPENAI_STREAMING_DONE_MARKER = "[DONE]"  # Ref: https://platform.openai.com/docs/api-reference/chat/create#stream


#: Retry on OpenAI errors
openai_retry = tenacity.retry(
    reraise=True,
    retry=tenacity.retry_if_exception_type(OpenAIError)
    and tenacity.retry_if_not_exception_type(OpenAIUnauthorizedError),
    wait=tenacity.wait_exponential(multiplier=OPENAI_BACKOFF),
    stop=tenacity.stop_after_attempt(OPENAI_MAX_RETRIES),
)


@openai_retry
def complete(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Query ChatGPT without streaming the response.

    :param url: The URL to query.
    :param headers: The headers to send with the request.
    :param payload: The payload to send with the request.
    :return: A list of strings containing the response from the OpenAI API.
    """
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=OPENAI_TIMEOUT)
    raise_for_status(response=response)
    json_response = json.loads(response.text)
    check_truncated_answers(result=json_response, payload=payload)
    metadata = [
        {
            "model": json_response.get("model", None),
            "index": choice.get("index", None),
            "finish_reason": choice.get("finish_reason", None),
            **json_response.get("usage", {}),
        }
        for choice in json_response.get("choices", [])
    ]
    replies = [choice["message"]["content"].strip() for choice in json_response.get("choices", [])]
    return replies, metadata


@openai_retry
def complete_stream(
    url: str, headers: Dict[str, str], payload: Dict[str, Any], callback: Callable
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Query ChatGPT and streams the response. Once the stream finishes, returns a list of strings just like
    self._query_llm()

    :param url: The URL to query.
    :param headers: The headers to send with the request.
    :param payload: The payload to send with the request.
    :param callback: A callback function that is called when a new token is received from the stream.
        The callback function should accept two parameters: the token received from the stream and **kwargs.
        The callback function should return the token that will be returned at the end of the streaming.
    :return: A list of strings containing the response from the OpenAI API.
    """
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=OPENAI_TIMEOUT, stream=True)
    raise_for_status(response=response)

    client = sseclient.SSEClient(response)  # type: ignore
    event_data = None
    tokens = []
    try:
        for event in client.events():
            if event.data == OPENAI_STREAMING_DONE_MARKER:
                break
            event_data = json.loads(event.data)
            delta = event_data["choices"][0]["delta"]
            token = delta["content"] if "content" in delta else None
            if token:
                tokens.append(callback(token, event_data=event_data["choices"]))
    finally:
        client.close()
    metadata = (
        [
            {
                "model": event_data.get("model", None),
                "index": choice.get("index", None),
                "finish_reason": choice.get("finish_reason", None),
            }
            for choice in event_data.get("choices", [])
        ]
        if event_data
        else []
    )
    return ["".join(tokens)], metadata


def raise_for_status(response: requests.Response):
    """
    Raises the appropriate OpenAI error in case of a bad response.

    :param response: The response returned from the OpenAI API.
    :raises OpenAIError: If the response status code is not 200.
    """
    if response.status_code >= 400:
        if response.status_code == 429:
            raise OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
        if response.status_code == 401:
            raise OpenAIUnauthorizedError(f"API key is invalid: {response.text}")
        raise OpenAIError(
            f"OpenAI returned an error.\n" f"Status code: {response.status_code}\n" f"Response body: {response.text}",
            status_code=response.status_code,
        )


def check_truncated_answers(result: Dict[str, Any], payload: Dict[str, Any]):
    """
    Check the `finish_reason` the answers returned by OpenAI completions endpoint.
    If the `finish_reason` is `length`, log a warning to the user.

    :param result: The result returned from the OpenAI API.
    :param payload: The payload sent to the OpenAI API.
    """
    truncated_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "length")
    if truncated_completions > 0:
        logger.warning(
            "%s out of the %s completions have been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions.",
            truncated_completions,
            payload["n"],
        )


def enforce_token_limit(prompt: str, tokenizer: "tiktoken.Encoding", max_tokens_limit: int) -> str:
    """
    Ensure that the length of the prompt is within the max tokens limit of the model.
    If needed, truncate the prompt text so that it fits within the limit.

    :param prompt: Prompt text to be sent to the generative model.
    :param tokenizer: The tokenizer used to encode the prompt.
    :param max_tokens_limit: The max tokens limit of the model.
    :return: The prompt text that fits within the max tokens limit of the model.
    """
    tiktoken_import.check()
    tokens = tokenizer.encode(prompt)
    tokens_count = len(tokens)
    if tokens_count > max_tokens_limit:
        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens to fit within the max token limit. "
            "Reduce the length of the prompt to prevent it from being cut off.",
            tokens_count,
            max_tokens_limit,
        )
        prompt = tokenizer.decode(tokens[:max_tokens_limit])
    return prompt


def enforce_token_limit_chat(
    chat: List[ChatMessage], tokenizer: "tiktoken.Encoding", max_tokens_limit: int, tokens_per_message_overhead: int
) -> List[ChatMessage]:
    """
    Ensure that the length of the chat is within the max tokens limit of the model.
    If needed, truncate the messages so that the chat fits within the limit.

    :param chat: The chat messages to be sent to the generative model.
    :param tokenizer: The tokenizer used to encode the chat.
    :param max_tokens_limit: The max tokens limit of the model.
    :param tokens_per_message_overhead: The number of tokens that are added to the prompt text for each message.
    :return: A chat that fits within the max tokens limit of the model.
    """
    messages_len = [len(tokenizer.encode(message.content)) + tokens_per_message_overhead for message in chat]
    if (total_chat_length := sum(messages_len)) <= max_tokens_limit:
        return chat

    logger.warning(
        "The chat have been truncated from %s tokens to %s tokens to fit within the max token limit. "
        "Reduce the length of the chat to prevent it from being cut off.",
        total_chat_length,
        max_tokens_limit,
    )
    cut_messages = []
    cut_messages_len: List[int] = []
    for message, message_len in zip(chat, messages_len):
        if sum(cut_messages_len) + message_len <= max_tokens_limit:
            cut_messages.append(message)
            cut_messages_len.append(message_len)
        else:
            remaining_tokens = max_tokens_limit - sum(cut_messages_len)
            cut_messages.append(
                ChatMessage(
                    content=enforce_token_limit(message.content, tokenizer, remaining_tokens), role=message.role
                )
            )
            break
    return cut_messages
