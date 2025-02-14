# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "HuggingFaceLocalGenerator",
    "HuggingFaceAPIGenerator",
    "OpenAIGenerator",
    "AzureOpenAIGenerator",
    "DALLEImageGenerator",
]


def HuggingFaceLocalGenerator():  # noqa: D103
    from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator

    return HuggingFaceLocalGenerator


def HuggingFaceAPIGenerator():  # noqa: D103
    from haystack.components.generators.hugging_face_api import HuggingFaceAPIGenerator

    return HuggingFaceAPIGenerator


def OpenAIGenerator():  # noqa: D103
    from haystack.components.generators.openai import (  # noqa: I001 (otherwise we end up with partial imports)
        OpenAIGenerator,
    )

    return OpenAIGenerator


def AzureOpenAIGenerator():  # noqa: D103
    from haystack.components.generators.azure import AzureOpenAIGenerator

    return AzureOpenAIGenerator


def DALLEImageGenerator():  # noqa: D103
    from haystack.components.generators.openai_dalle import DALLEImageGenerator

    return DALLEImageGenerator
