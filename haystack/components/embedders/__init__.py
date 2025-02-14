# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "AzureOpenAIDocumentEmbedder",
    "AzureOpenAITextEmbedder",
    "HuggingFaceAPIDocumentEmbedder",
    "HuggingFaceAPITextEmbedder",
    "OpenAIDocumentEmbedder",
    "OpenAITextEmbedder",
    "SentenceTransformersDocumentEmbedder",
    "SentenceTransformersTextEmbedder",
]


def AzureOpenAIDocumentEmbedder():  # noqa: D103
    from haystack.components.embedders.azure_document_embedder import AzureOpenAIDocumentEmbedder

    return AzureOpenAIDocumentEmbedder


def AzureOpenAITextEmbedder():  # noqa: D103
    from haystack.components.embedders.azure_text_embedder import AzureOpenAITextEmbedder

    return AzureOpenAITextEmbedder


def HuggingFaceAPIDocumentEmbedder():  # noqa: D103
    from haystack.components.embedders.hugging_face_api_document_embedder import HuggingFaceAPIDocumentEmbedder

    return HuggingFaceAPIDocumentEmbedder


def HuggingFaceAPITextEmbedder():  # noqa: D103
    from haystack.components.embedders.hugging_face_api_text_embedder import HuggingFaceAPITextEmbedder

    return HuggingFaceAPITextEmbedder


def OpenAIDocumentEmbedder():  # noqa: D103
    from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder

    return OpenAIDocumentEmbedder


def OpenAITextEmbedder():  # noqa: D103
    from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder

    return OpenAITextEmbedder


def SentenceTransformersDocumentEmbedder():  # noqa: D103
    from haystack.components.embedders.sentence_transformers_document_embedder import (
        SentenceTransformersDocumentEmbedder,
    )

    return SentenceTransformersDocumentEmbedder


def SentenceTransformersTextEmbedder():  # noqa: D103
    from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder

    return SentenceTransformersTextEmbedder
