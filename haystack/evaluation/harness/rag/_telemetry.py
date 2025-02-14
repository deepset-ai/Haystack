# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from haystack.telemetry._telemetry import send_telemetry

from .parameters import RAGEvaluationInput, RAGEvaluationMetric, RAGEvaluationOverrides

if TYPE_CHECKING:
    from .harness import DefaultRAGArchitecture, RAGEvaluationHarness


@dataclass
class TelemetryPayload:  # pylint: disable=too-many-instance-attributes
    """
    Represents the telemetry payload for evaluating a RAG model.

    :param eval_metrics:
        Active evaluation metrics and per-metric metadata.
    :param num_queries:
        Number of queries used for evaluation.
    :param execution_time_sec:
        Execution time in seconds for the evaluation.
    :param default_architecture:
        Default RAG architecture used for the RAG pipeline.
    :param num_gt_answers:
        Number of ground truth answers used in evaluation.
    :param num_gt_contexts:
        Number of ground truth contexts used in evaluation.
    :param rag_pipeline_overrides:
        Indicates if the RAG pipeline has any overrides.
    :param eval_pipeline_overrides:
        Indicates if the evaluation pipeline has any overrides.
    """

    eval_metrics: Dict[RAGEvaluationMetric, Optional[Dict[str, Any]]]
    num_queries: int
    execution_time_sec: float

    default_architecture: Optional["DefaultRAGArchitecture"] = None
    num_gt_answers: Optional[int] = None
    num_gt_contexts: Optional[int] = None
    rag_pipeline_overrides: Optional[bool] = None
    eval_pipeline_overrides: Optional[bool] = None

    def serialize(self) -> Dict[str, Any]:
        out = asdict(self)

        out["eval_metrics"] = {k.value: v for k, v in self.eval_metrics.items()}
        out["default_architecture"] = self.default_architecture.value if self.default_architecture else None

        return out


@send_telemetry
def harness_eval_run_complete(
    harness: "RAGEvaluationHarness",
    inputs: RAGEvaluationInput,
    execution_time_sec: float,
    overrides: Optional[RAGEvaluationOverrides] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    payload = harness._telemetry_payload

    payload = replace(
        payload,
        num_queries=len(inputs.queries),
        execution_time_sec=execution_time_sec,
        num_gt_answers=(len(inputs.ground_truth_answers) if inputs.ground_truth_answers else None),
        num_gt_contexts=(len(inputs.ground_truth_documents) if inputs.ground_truth_documents else None),
        rag_pipeline_overrides=(overrides.rag_pipeline is not None if overrides else None),
        eval_pipeline_overrides=(overrides.eval_pipeline is not None if overrides else None),
    )

    return "RAG evaluation harness eval run", payload.serialize()
