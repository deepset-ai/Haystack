# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "HuggingFaceLocalChatGenerator",
    "HuggingFaceAPIChatGenerator",
    "OpenAIChatGenerator",
    "AzureOpenAIChatGenerator",
]


def AzureOpenAIChatGenerator():  # noqa: D103
    from haystack.components.generators.chat.azure import AzureOpenAIChatGenerator

    return AzureOpenAIChatGenerator


def HuggingFaceLocalChatGenerator():  # noqa: D103
    from haystack.components.generators.chat.hugging_face_local import HuggingFaceLocalChatGenerator

    return HuggingFaceLocalChatGenerator


def HuggingFaceAPIChatGenerator():  # noqa: D103
    from haystack.components.generators.chat.hugging_face_api import HuggingFaceAPIChatGenerator

    return HuggingFaceAPIChatGenerator


def OpenAIChatGenerator():  # noqa: D103
    from haystack.components.generators.chat.openai import (  # noqa: I001 (otherwise we end up with partial imports)
        OpenAIChatGenerator,
    )

    return OpenAIChatGenerator
