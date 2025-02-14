# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["AnswerBuilder", "PromptBuilder", "ChatPromptBuilder"]


def AnswerBuilder():  # noqa: D103
    from haystack.components.builders.answer_builder import AnswerBuilder

    return AnswerBuilder


def ChatPromptBuilder():  # noqa: D103
    from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder

    return ChatPromptBuilder


def PromptBuilder():  # noqa: D103
    from haystack.components.builders.prompt_builder import PromptBuilder

    return PromptBuilder
