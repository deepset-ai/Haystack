# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FilterRetriever", "InMemoryEmbeddingRetriever", "InMemoryBM25Retriever", "SentenceWindowRetriever"]


def FilterRetriever():  # noqa: D103
    from haystack.components.retrievers.filter_retriever import FilterRetriever

    return FilterRetriever


def InMemoryBM25Retriever():  # noqa: D103
    from haystack.components.retrievers.in_memory.bm25_retriever import InMemoryBM25Retriever

    return InMemoryBM25Retriever


def InMemoryEmbeddingRetriever():  # noqa: D103
    from haystack.components.retrievers.in_memory.embedding_retriever import InMemoryEmbeddingRetriever

    return InMemoryEmbeddingRetriever


def SentenceWindowRetriever():  # noqa: D103
    from haystack.components.retrievers.sentence_window_retriever import SentenceWindowRetriever

    return SentenceWindowRetriever
