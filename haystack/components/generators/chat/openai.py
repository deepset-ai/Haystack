# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.openai_utils import _convert_message_to_openai_format
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable

logger = logging.getLogger(__name__)


@component
class OpenAIChatGenerator:
    """
    Completes chats using OpenAI's large language models (LLMs).

    It works with the gpt-4 and gpt-3.5-turbo models and supports streaming responses
    from OpenAI API. It uses [ChatMessage](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
    format in input and output.

    You can customize how the text is generated by passing parameters to the
    OpenAI API. Use the `**generation_kwargs` argument when you initialize
    the component or when you run it. Any parameter that works with
    `openai.ChatCompletion.create` will work here too.

    For details on OpenAI API parameters, see
    [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat).

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OpenAIChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    Output:
    ```
    {'replies':
        [ChatMessage(content='Natural Language Processing (NLP) is a branch of artificial intelligence
            that focuses on enabling computers to understand, interpret, and generate human language in
            a way that is meaningful and useful.',
         role=<ChatRole.ASSISTANT: 'assistant'>, name=None,
         meta={'model': 'gpt-4o-mini', 'index': 0, 'finish_reason': 'stop',
         'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})
        ]
    }
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-4o-mini",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Creates an instance of OpenAIChatGenerator. Unless specified otherwise in `model`, uses OpenAI's gpt-4o-mini

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key: The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
            as an argument.
        :param api_base_url: An optional base URL.
        :param organization: Your organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param generation_kwargs: Other parameters to use for the model. These parameters are sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. For example, 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.organization = organization

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", 5))

        self.client = OpenAI(
            api_key=api_key.resolve_value(),
            organization=organization,
            base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
                                  override the parameters passed during component initialization.
                                  For details on OpenAI API parameters, see
                                  [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).

        :returns:
            A list containing the generated responses as ChatMessage instances.
        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self.streaming_callback

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [_convert_message_to_openai_format(message) for message in messages]

        chat_completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore # openai expects list of specific message types
            stream=streaming_callback is not None,
            **generation_kwargs,
        )

        completions: List[ChatMessage] = []
        # if streaming is enabled, the completion is a Stream of ChatCompletionChunk
        if isinstance(chat_completion, Stream):
            num_responses = generation_kwargs.pop("n", 1)
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")
            chunks: List[StreamingChunk] = []
            chunk = None

            # pylint: disable=not-an-iterable
            for chunk in chat_completion:
                if chunk.choices and streaming_callback:
                    chunk_delta: StreamingChunk = self._build_chunk(chunk)
                    chunks.append(chunk_delta)
                    streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
            completions = [self._connect_chunks(chunk, chunks)]
        # if streaming is disabled, the completion is a ChatCompletion
        elif isinstance(chat_completion, ChatCompletion):
            completions = [self._build_message(chat_completion, choice) for choice in chat_completion.choices]

        # before returning, do post-processing of the completions
        for message in completions:
            self._check_finish_reason(message)

        return {"replies": completions}

    def _connect_chunks(self, chunk: Any, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.

        :param chunk: The last chunk returned by the OpenAI API.
        :param chunks: The list of all chunks returned by the OpenAI API.
        """
        is_tools_call = bool(chunks[0].meta.get("tool_calls"))
        is_function_call = bool(chunks[0].meta.get("function_call"))
        # if it's a tool call or function call, we need to build the payload dict from all the chunks
        if is_tools_call or is_function_call:
            tools_len = 1 if is_function_call else len(chunks[0].meta.get("tool_calls", []))
            # don't change this approach of building payload dicts, otherwise mypy will complain
            p_def: Dict[str, Any] = {
                "index": 0,
                "id": "",
                "function": {"arguments": "", "name": ""},
                "type": "function",
            }
            payloads = [copy.deepcopy(p_def) for _ in range(tools_len)]
            for chunk_payload in chunks:
                if is_tools_call:
                    deltas = chunk_payload.meta.get("tool_calls") or []
                else:
                    deltas = [chunk_payload.meta["function_call"]] if chunk_payload.meta.get("function_call") else []

                # deltas is a list of ChoiceDeltaToolCall or ChoiceDeltaFunctionCall
                for i, delta in enumerate(deltas):
                    payload = payloads[i]
                    if is_tools_call:
                        payload["id"] = delta.id or payload["id"]
                        payload["type"] = delta.type or payload["type"]
                        if delta.function:
                            payload["function"]["name"] += delta.function.name or ""
                            payload["function"]["arguments"] += delta.function.arguments or ""
                    elif is_function_call:
                        payload["function"]["name"] += delta.name or ""
                        payload["function"]["arguments"] += delta.arguments or ""
            complete_response = ChatMessage.from_assistant(json.dumps(payloads))
        else:
            complete_response = ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))
        complete_response.meta.update(
            {
                "model": chunk.model,
                "index": 0,
                "finish_reason": chunk.choices[0].finish_reason,
                "usage": {},  # we don't have usage data for streaming responses
            }
        )
        return complete_response

    def _build_message(self, completion: ChatCompletion, choice: Choice) -> ChatMessage:
        """
        Converts the non-streaming response from the OpenAI API to a ChatMessage.

        :param completion: The completion returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The ChatMessage.
        """
        message: ChatCompletionMessage = choice.message
        content = message.content or ""
        if message.function_call:
            # here we mimic the tools format response so that if user passes deprecated `functions` parameter
            # she'll get the same output as if new `tools` parameter was passed
            # use pydantic model dump to serialize the function call
            content = json.dumps(
                [{"function": message.function_call.model_dump(), "type": "function", "id": completion.id}]
            )
        elif message.tool_calls:
            # new `tools` parameter was passed, use pydantic model dump to serialize the tool calls
            content = json.dumps([tc.model_dump() for tc in message.tool_calls])

        chat_message = ChatMessage.from_assistant(content)
        chat_message.meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage or {}),
            }
        )
        return chat_message

    def _build_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk:
        """
        Converts the streaming response chunk from the OpenAI API to a StreamingChunk.

        :param chunk: The chunk returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The StreamingChunk.
        """
        # we stream the content of the chunk if it's not a tool or function call
        choice: ChunkChoice = chunk.choices[0]
        content = choice.delta.content or ""
        chunk_message = StreamingChunk(content)
        # but save the tool calls and function call in the meta if they are present
        # and then connect the chunks in the _connect_chunks method
        chunk_message.meta.update(
            {
                "model": chunk.model,
                "index": choice.index,
                "tool_calls": choice.delta.tool_calls,
                "function_call": choice.delta.function_call,
                "finish_reason": choice.finish_reason,
            }
        )
        return chunk_message

    def _check_finish_reason(self, message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.

        If the `finish_reason` is `length` or `content_filter`, log a warning.
        :param message: The message returned by the LLM.
        """
        if message.meta["finish_reason"] == "length":
            logger.warning(
                "The completion for index {index} has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
        if message.meta["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index {index} has been truncated due to the content filter.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
