# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Callable, Dict, Set

from haystack import Pipeline
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    FaithfulnessEvaluator,
    SASEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode

from .parameters import RAGEvaluationMetric


def default_rag_evaluation_pipeline(metrics: Set[RAGEvaluationMetric]) -> Pipeline:
    """
    Builds the default evaluation pipeline for RAG.

    :param metrics:
        The set of metrics to include in the pipeline.
    :returns:
        The evaluation pipeline.
    """
    pipeline = Pipeline()

    metric_ctors: Dict[RAGEvaluationMetric, Callable] = {
        RAGEvaluationMetric.DOCUMENT_MAP: DocumentMAPEvaluator,
        RAGEvaluationMetric.DOCUMENT_MRR: DocumentMRREvaluator,
        RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT: partial(DocumentRecallEvaluator, mode=RecallMode.SINGLE_HIT),
        RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT: partial(DocumentRecallEvaluator, mode=RecallMode.MULTI_HIT),
        RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY: partial(
            SASEvaluator, model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        RAGEvaluationMetric.FAITHFULNESS: partial(FaithfulnessEvaluator, raise_on_failure=False),
        RAGEvaluationMetric.CONTEXT_RELEVANCE: partial(ContextRelevanceEvaluator, raise_on_failure=False),
    }

    for metric in metrics:
        ctor = metric_ctors[metric]
        pipeline.add_component(metric.value, ctor())

    return pipeline
