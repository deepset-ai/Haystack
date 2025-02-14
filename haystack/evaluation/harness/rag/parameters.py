# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from haystack import Document
from haystack.evaluation.eval_run_result import EvaluationRunResult


class RAGExpectedComponent(Enum):
    """
    Represents the basic components in a RAG pipeline that are, by default, required to be present for evaluation.

    These can be separate components in a pipeline or a single component that performs
    multiple tasks.
    """

    #: The component in a RAG pipeline that accepts the user query.
    #: Expected inputs: `query` - Name of input that contains the query string.
    QUERY_PROCESSOR = "query_processor"

    #: The component in a RAG pipeline that retrieves documents based on the query.
    #: Expected outputs: `retrieved_documents` - Name of output containing retrieved documents.
    DOCUMENT_RETRIEVER = "document_retriever"

    #: The component in a RAG pipeline that generates responses based on the query and the retrieved documents.
    #: Can be optional if the harness is only evaluating retrieval.
    #: Expected outputs: `replies` - Name of out containing the LLM responses. Only the first response is used.
    RESPONSE_GENERATOR = "response_generator"


@dataclass(frozen=True)
class RAGExpectedComponentMetadata:
    """
    Metadata for a `RAGExpectedComponent`.

    :param name:
        Name of the component in the pipeline.
    :param input_mapping:
        Mapping of the expected inputs to
        corresponding component input names.
    :param output_mapping:
        Mapping of the expected outputs to
        corresponding component output names.
    """

    name: str
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)


class RAGEvaluationMetric(Enum):
    """
    Represents the metrics that can be used to evaluate a RAG pipeline.
    """

    #: Document Mean Average Precision.
    #: Required RAG components: Query Processor, Document Retriever.
    DOCUMENT_MAP = "metric_doc_map"

    #: Document Mean Reciprocal Rank.
    #: Required RAG components: Query Processor, Document Retriever.
    DOCUMENT_MRR = "metric_doc_mrr"

    #: Document Recall with a single hit.
    #: Required RAG components: Query Processor, Document Retriever.
    DOCUMENT_RECALL_SINGLE_HIT = "metric_doc_recall_single"

    #: Document Recall with multiple hits.
    #: Required RAG components: Query Processor, Document Retriever.
    DOCUMENT_RECALL_MULTI_HIT = "metric_doc_recall_multi"

    #: Semantic Answer Similarity.
    #: Required RAG components: Query Processor, Response Generator.
    SEMANTIC_ANSWER_SIMILARITY = "metric_sas"

    #: Faithfulness.
    #: Required RAG components: Query Processor, Document Retriever, Response Generator.
    FAITHFULNESS = "metric_faithfulness"

    #: Context Relevance.
    #: Required RAG components: Query Processor, Document Retriever.
    CONTEXT_RELEVANCE = "metric_context_relevance"


@dataclass(frozen=True)
class RAGEvaluationInput:
    """
    Input passed to the RAG evaluation harness.

    :param queries:
        The queries passed to the RAG pipeline.
    :param ground_truth_documents:
        The ground truth documents passed to the
        evaluation pipeline. Only required for metrics
        that need the ground truth documents. Corresponds to the queries.
    :param ground_truth_answers:
        The ground truth answers passed to the
        evaluation pipeline. Only required for metrics
        that need the ground truth answeres. Corresponds to the queries.
    :param rag_pipeline_inputs:
        Additional inputs to pass to the RAG pipeline. Each
        key is the name of the component, and its value is a dictionary
        with the input name and a list of values, each corresponding
        to a query.
    """

    queries: List[str]
    ground_truth_documents: Optional[List[List[Document]]] = None
    ground_truth_answers: Optional[List[str]] = None
    rag_pipeline_inputs: Optional[Dict[str, Dict[str, List[Any]]]] = None


@dataclass(frozen=True)
class RAGEvaluationOverrides:
    """
    Overrides for a RAG evaluation run.

    Used to override the init parameters of components in
    either or both the evaluated and evaluation pipelines.

    :param rag_pipeline:
        Overrides for the RAG pipeline. Each
        key is a component name, and its value is a dictionary
        with init parameters to override.
    :param eval_pipeline:
        Overrides for the evaluation pipeline. Each
        key is a RAG metric, and its value is a dictionary
        with init parameters to override.
    """

    rag_pipeline: Optional[Dict[str, Dict[str, Any]]] = None
    eval_pipeline: Optional[Dict[RAGEvaluationMetric, Dict[str, Any]]] = None


@dataclass(frozen=True)
class RAGEvaluationOutput:
    """
    Represents the output of a RAG evaluation run.

    :param evaluated_pipeline:
        A serialized version of the evaluated pipeline, including overrides.
    :param evaluation_pipeline:
        A serialized version of the evaluation pipeline, including overrides.
    :param inputs:
        Input passed to the evaluation harness.
    :param results:
        Results of the evaluation run.
    """

    evaluated_pipeline: str
    evaluation_pipeline: str
    inputs: RAGEvaluationInput
    results: EvaluationRunResult
