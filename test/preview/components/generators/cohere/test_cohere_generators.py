import os
from unittest.mock import patch, Mock
from copy import deepcopy

import pytest
import cohere
from cohere.responses.generation import StreamingText

from haystack.preview.components.generators.cohere.cohere import CohereGenerator
from haystack.preview.components.generators.cohere.cohere import default_streaming_callback


class TestGPTGenerator:
    @pytest.mark.unit
    def test_init_default(self):
        component = CohereGenerator(api_key="test-api-key")
        assert component.api_key == "test-api-key"
        assert component.model == "command"
        assert component.streaming_callback is None
        assert component.api_base_url == cohere.COHERE_API_URL
        assert component.model_parameters == {}

    @pytest.mark.unit
    def test_init_with_parameters(self):
        callback = lambda x: x
        component = CohereGenerator(
            api_key="test-api-key",
            model="command-light",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=callback,
            api_base_url="test-base-url",
        )
        assert component.api_key == "test-api-key"
        assert component.model == "command-light"
        assert component.streaming_callback == callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.unit
    def test_to_dict_default(self):
        component = CohereGenerator(api_key="test-api-key")
        data = component.to_dict()
        assert data == {
            "type": "CohereGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model": "command",
                "streaming_callback": None,
                "api_base_url": cohere.COHERE_API_URL,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_parameters(self):
        component = CohereGenerator(
            api_key="test-api-key",
            model="command-light",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=default_streaming_callback,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "CohereGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model": "command-light",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.cohere.cohere.default_streaming_callback",
            },
        }
    @pytest.mark.unit
    def test_to_dict_with_lambda_streaming_callback(self):
        component = CohereGenerator(
            api_key="test-api-key",
            model="command",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            'type': 'CohereGenerator', 
            'init_parameters': {
                'api_key': 'test-api-key', 
                'model': 'command', 
                'streaming_callback': 'test_cohere_generators.<lambda>', 
                'api_base_url': 'test-base-url', 
                'max_tokens': 10, 
                'some_test_param': 'test-params'
                }
            }
    @pytest.mark.unit
    def test_from_dict(self):
        data = {
            "type": "CohereGenerator",
            "init_parameters": {
                "api_key": "test-api-key",
                "model": "command",
                "max_tokens": 10,
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.preview.components.generators.cohere.cohere.default_streaming_callback",
            },
        }
        component = CohereGenerator.from_dict(data)
        assert component.api_key == "test-api-key"
        assert component.model == "command"
        assert component.streaming_callback == default_streaming_callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    # @pytest.mark.unit
    # def test_run_with_parameters(self):
    #     with patch("haystack.preview.components.generators.openai.gpt.openai.ChatCompletion") as cohere_patch:
    #         cohere_patch.create.side_effect = mock_openai_response
    #         component = CohereGenerator(api_key="test-api-key", max_tokens=10)
    #         component.run(prompt="test-prompt-1")
    #         gpt_patch.create.assert_called_once_with(
    #             model="gpt-3.5-turbo",
    #             api_key="test-api-key",
    #             messages=[{"role": "user", "content": "test-prompt-1"}],
    #             stream=False,
    #             max_tokens=10,
    #         )

    @pytest.mark.unit
    def test_check_truncated_answers(self, caplog):
        component = CohereGenerator(api_key="test-api-key")
        metadata = [
            {"finish_reason": "MAX_TOKENS"},
        ]
        component._check_truncated_answers(metadata)
        assert caplog.records[0].message == (
            "Responses have been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )
    
    @pytest.mark.skipif(
        not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run(self):
        component = CohereGenerator(api_key=os.environ.get("CO_API_KEY"))
        results = component.run(prompt="What's the capital of France?")
        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["metadata"]) == 1
        assert results["metadata"][0]["finish_reason"] == "COMPLETE"

    @pytest.mark.skipif(
        not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run_wrong_model_name(self):
        component =  CohereGenerator(model="something-obviously-wrong", api_key=os.environ.get("CO_API_KEY"))
        with pytest.raises(cohere.CohereAPIError, match="model not found, make sure the correct model ID was used and that you have access to the model."):
            component.run(prompt="What's the capital of France?")

    @pytest.mark.skipif(
        not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""

            def __call__(self, chunk):
                self.responses += chunk.text
                return chunk

        callback = Callback()
        component = CohereGenerator(os.environ.get("CO_API_KEY"), streaming_callback=callback)
        results = component.run(prompt="What's the capital of France?")

        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["metadata"]) == 1
        assert results["metadata"][0]["finish_reason"] == "COMPLETE"
        assert callback.responses == results["replies"][0]
