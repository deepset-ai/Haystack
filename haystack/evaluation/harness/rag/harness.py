# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import time
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from haystack import Pipeline
from haystack.evaluation.eval_run_result import EvaluationRunResult

from ...util.helpers import aggregate_batched_pipeline_outputs, deaggregate_batched_pipeline_inputs
from ...util.pipeline_pair import PipelinePair
from ..evaluation_harness import EvaluationHarness
from ._telemetry import TelemetryPayload, harness_eval_run_complete
from .evaluation_pipeline import default_rag_evaluation_pipeline
from .parameters import (
    RAGEvaluationInput,
    RAGEvaluationMetric,
    RAGEvaluationOutput,
    RAGEvaluationOverrides,
    RAGExpectedComponent,
    RAGExpectedComponentMetadata,
)


class DefaultRAGArchitecture(Enum):
    """
    Represents default RAG pipeline architectures that can be used with the evaluation harness.
    """

    #: A RAG pipeline with:
    #: - A query embedder component named 'query_embedder' with a 'text' input.
    #: - A document retriever component named 'retriever' with a 'documents' output.
    EMBEDDING_RETRIEVAL = "embedding_retrieval"

    #: A RAG pipeline with:
    #: - A document retriever component named 'retriever' with a 'query' input and a 'documents' output.
    KEYWORD_RETRIEVAL = "keyword_retrieval"

    #: A RAG pipeline with:
    #: - A query embedder component named 'query_embedder' with a 'text' input.
    #: - A document retriever component named 'retriever' with a 'documents' output.
    #: - A response generator component named 'generator' with a 'replies' output.
    GENERATION_WITH_EMBEDDING_RETRIEVAL = "generation_with_embedding_retrieval"

    #: A RAG pipeline with:
    #: - A document retriever component named 'retriever' with a 'query' input and a 'documents' output.
    #: - A response generator component named 'generator' with a 'replies' output.
    GENERATION_WITH_KEYWORD_RETRIEVAL = "generation_with_keyword_retrieval"

    @property
    def expected_components(self) -> Dict[RAGExpectedComponent, RAGExpectedComponentMetadata]:
        """
        Returns the expected components for the architecture.

        :returns:
            The expected components.
        """
        if self in (
            DefaultRAGArchitecture.EMBEDDING_RETRIEVAL,
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
        ):
            expected = {
                RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                    name="query_embedder", input_mapping={"query": "text"}
                ),
                RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                    name="retriever", output_mapping={"retrieved_documents": "documents"}
                ),
            }
        elif self in (
            DefaultRAGArchitecture.KEYWORD_RETRIEVAL,
            DefaultRAGArchitecture.GENERATION_WITH_KEYWORD_RETRIEVAL,
        ):
            expected = {
                RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                    name="retriever", input_mapping={"query": "query"}
                ),
                RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                    name="retriever", output_mapping={"retrieved_documents": "documents"}
                ),
            }
        else:
            raise NotImplementedError(f"Unexpected default RAG architecture: {self}")

        if self in (
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
            DefaultRAGArchitecture.GENERATION_WITH_KEYWORD_RETRIEVAL,
        ):
            expected[RAGExpectedComponent.RESPONSE_GENERATOR] = RAGExpectedComponentMetadata(
                name="generator", output_mapping={"replies": "replies"}
            )

        return expected


class RAGEvaluationHarness(EvaluationHarness[RAGEvaluationInput, RAGEvaluationOverrides, RAGEvaluationOutput]):
    """
    Evaluation harness for evaluating RAG pipelines.
    """

    def __init__(
        self,
        rag_pipeline: Pipeline,
        rag_components: Union[DefaultRAGArchitecture, Dict[RAGExpectedComponent, RAGExpectedComponentMetadata]],
        metrics: Set[RAGEvaluationMetric],
        *,
        progress_bar: bool = True,
    ):
        """
        Create an evaluation harness for evaluating basic RAG pipelines.

        :param rag_pipeline:
            The RAG pipeline to evaluate.
        :param rag_components:
            Either a default RAG architecture or a mapping
            of expected components to their metadata.
        :param metrics:
            The metrics to use during evaluation.
        :param progress_bar:
            Whether to display a progress bar during evaluation.
        """
        super().__init__()

        self._telemetry_payload = TelemetryPayload(
            eval_metrics={m: None for m in metrics},
            num_queries=0,
            execution_time_sec=0.0,
            default_architecture=(rag_components if isinstance(rag_components, DefaultRAGArchitecture) else None),
        )

        if isinstance(rag_components, DefaultRAGArchitecture):
            rag_components = rag_components.expected_components

        self._validate_rag_components(rag_pipeline, rag_components, metrics)

        self.rag_pipeline = rag_pipeline
        self.rag_components = deepcopy(rag_components)
        self.metrics = deepcopy(metrics)
        self.evaluation_pipeline = default_rag_evaluation_pipeline(metrics)
        self.progress_bar = progress_bar

    def run(  # noqa: D102
        self,
        inputs: RAGEvaluationInput,
        *,
        overrides: Optional[RAGEvaluationOverrides] = None,
        run_name: Optional[str] = "RAG Evaluation",
    ) -> RAGEvaluationOutput:
        start_time = time.time()

        rag_inputs = self._prepare_rag_pipeline_inputs(inputs)
        eval_inputs = self._prepare_eval_pipeline_additional_inputs(inputs)
        pipeline_pair = self._generate_eval_run_pipelines(overrides)

        pipeline_outputs = pipeline_pair.run_first_as_batch(rag_inputs, eval_inputs, progress_bar=self.progress_bar)
        rag_outputs, eval_outputs = (pipeline_outputs["first"], pipeline_outputs["second"])

        result_inputs: Dict[str, List[Any]] = {
            "questions": inputs.queries,
            "contexts": [
                [doc.content for doc in docs]
                for docs in self._lookup_component_output(
                    RAGExpectedComponent.DOCUMENT_RETRIEVER, rag_outputs, "retrieved_documents"
                )
            ],
        }
        if RAGExpectedComponent.RESPONSE_GENERATOR in self.rag_components:
            result_inputs["responses"] = self._lookup_component_output(
                RAGExpectedComponent.RESPONSE_GENERATOR, rag_outputs, "replies"
            )

        if inputs.ground_truth_answers is not None:
            result_inputs["ground_truth_answers"] = inputs.ground_truth_answers
        if inputs.ground_truth_documents is not None:
            result_inputs["ground_truth_documents"] = [
                [doc.content for doc in docs] for docs in inputs.ground_truth_documents
            ]

        assert run_name is not None
        run_results = EvaluationRunResult(run_name, inputs=result_inputs, results=eval_outputs)

        harness_eval_run_complete(self, inputs, time.time() - start_time, overrides)

        return RAGEvaluationOutput(
            evaluated_pipeline=pipeline_pair.first.dumps(),
            evaluation_pipeline=pipeline_pair.second.dumps(),
            inputs=deepcopy(inputs),
            results=run_results,
        )

    def _lookup_component_output(
        self, component: RAGExpectedComponent, outputs: Dict[str, Dict[str, Any]], output_name: str
    ) -> Any:
        name = self.rag_components[component].name
        mapping = self.rag_components[component].output_mapping
        output_name = mapping[output_name]
        return outputs[name][output_name]

    def _generate_eval_run_pipelines(self, overrides: Optional[RAGEvaluationOverrides]) -> PipelinePair:
        if overrides is None:
            rag_overrides = None
            eval_overrides = None
        else:
            rag_overrides = overrides.rag_pipeline
            eval_overrides = overrides.eval_pipeline

        if eval_overrides is not None:
            for metric in eval_overrides.keys():
                if metric not in self.metrics:
                    raise ValueError(f"Cannot override parameters of unused evaluation metric '{metric.value}'")

            eval_overrides = {k.value: v for k, v in eval_overrides.items()}  # type: ignore

        rag_pipeline = self._override_pipeline(self.rag_pipeline, rag_overrides)
        eval_pipeline = self._override_pipeline(self.evaluation_pipeline, eval_overrides)  # type: ignore

        included_first_outputs = {self.rag_components[RAGExpectedComponent.DOCUMENT_RETRIEVER].name}
        if RAGExpectedComponent.RESPONSE_GENERATOR in self.rag_components:
            included_first_outputs.add(self.rag_components[RAGExpectedComponent.RESPONSE_GENERATOR].name)

        return PipelinePair(
            first=rag_pipeline,
            second=eval_pipeline,
            outputs_to_inputs=self._map_rag_eval_pipeline_io(),
            map_first_outputs=lambda x: self._aggregate_rag_outputs(  # pylint: disable=unnecessary-lambda
                x
            ),
            included_first_outputs=included_first_outputs,
            pre_execution_callback_first=lambda: print("Executing RAG pipeline..."),
            pre_execution_callback_second=lambda: print("Executing evaluation pipeline..."),
        )

    def _aggregate_rag_outputs(self, outputs: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        aggregate = aggregate_batched_pipeline_outputs(outputs)

        if RAGExpectedComponent.RESPONSE_GENERATOR in self.rag_components:
            # We only care about the first response from the generator.
            generator_name = self.rag_components[RAGExpectedComponent.RESPONSE_GENERATOR].name
            replies_output_name = self.rag_components[RAGExpectedComponent.RESPONSE_GENERATOR].output_mapping["replies"]
            aggregate[generator_name][replies_output_name] = [
                r[0] for r in aggregate[generator_name][replies_output_name]
            ]

        return aggregate

    def _map_rag_eval_pipeline_io(self) -> Dict[str, List[str]]:
        # We currently only have metric components in the eval pipeline.
        # So, we just map those inputs to the outputs of the rag pipeline.
        metric_inputs_to_component_outputs = {
            RAGEvaluationMetric.DOCUMENT_MAP: {
                "retrieved_documents": (RAGExpectedComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
            RAGEvaluationMetric.DOCUMENT_MRR: {
                "retrieved_documents": (RAGExpectedComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
            RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT: {
                "retrieved_documents": (RAGExpectedComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
            RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT: {
                "retrieved_documents": (RAGExpectedComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
            RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY: {
                "predicted_answers": (RAGExpectedComponent.RESPONSE_GENERATOR, "replies")
            },
            RAGEvaluationMetric.FAITHFULNESS: {
                "contexts": (RAGExpectedComponent.DOCUMENT_RETRIEVER, "retrieved_documents"),
                "predicted_answers": (RAGExpectedComponent.RESPONSE_GENERATOR, "replies"),
            },
            RAGEvaluationMetric.CONTEXT_RELEVANCE: {
                "contexts": (RAGExpectedComponent.DOCUMENT_RETRIEVER, "retrieved_documents")
            },
        }

        outputs_to_inputs: Dict[str, List[str]] = {}
        for metric in self.metrics:
            io = metric_inputs_to_component_outputs[metric]
            for metric_input_name, (component, component_output_name) in io.items():
                component_out = (
                    f"{self.rag_components[component].name}."
                    f"{self.rag_components[component].output_mapping[component_output_name]}"
                )
                metric_in = f"{metric.value}.{metric_input_name}"
                if component_out not in outputs_to_inputs:
                    outputs_to_inputs[component_out] = []
                outputs_to_inputs[component_out].append(metric_in)

        return outputs_to_inputs

    def _prepare_rag_pipeline_inputs(self, inputs: RAGEvaluationInput) -> List[Dict[str, Dict[str, Any]]]:
        query_embedder_name = self.rag_components[RAGExpectedComponent.QUERY_PROCESSOR].name
        query_embedder_text_input = self.rag_components[RAGExpectedComponent.QUERY_PROCESSOR].input_mapping["query"]

        if inputs.rag_pipeline_inputs is not None:
            # Ensure that the query embedder input is not provided as additional input.
            existing = inputs.rag_pipeline_inputs.get(query_embedder_name)
            if existing is not None:
                existing = existing.get(query_embedder_text_input)  # type: ignore
                if existing is not None:
                    raise ValueError(
                        f"Query embedder input '{query_embedder_text_input}' cannot be provided as additional input."
                    )

            # Add the queries as an aggregate input.
            rag_inputs = deepcopy(inputs.rag_pipeline_inputs)
            if query_embedder_name not in rag_inputs:
                rag_inputs[query_embedder_name] = {}
            rag_inputs[query_embedder_name][query_embedder_text_input] = deepcopy(inputs.queries)
        else:
            rag_inputs = {query_embedder_name: {query_embedder_text_input: deepcopy(inputs.queries)}}

        separate_rag_inputs = deaggregate_batched_pipeline_inputs(rag_inputs)
        return separate_rag_inputs

    def _prepare_eval_pipeline_additional_inputs(self, inputs: RAGEvaluationInput) -> Dict[str, Dict[str, Any]]:
        eval_inputs: Dict[str, Dict[str, List[Any]]] = {}

        for metric in self.metrics:
            if metric in (
                RAGEvaluationMetric.DOCUMENT_MAP,
                RAGEvaluationMetric.DOCUMENT_MRR,
                RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT,
                RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT,
            ):
                if inputs.ground_truth_documents is None:
                    raise ValueError(f"Ground truth documents required for metric '{metric.value}'.")
                if len(inputs.ground_truth_documents) != len(inputs.queries):
                    raise ValueError("Length of ground truth documents should match the number of queries.")

                eval_inputs[metric.value] = {"ground_truth_documents": inputs.ground_truth_documents}
            elif metric in (RAGEvaluationMetric.FAITHFULNESS, RAGEvaluationMetric.CONTEXT_RELEVANCE):
                eval_inputs[metric.value] = {"questions": inputs.queries}
            elif metric == RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY:
                if inputs.ground_truth_answers is None:
                    raise ValueError(f"Ground truth answers required for metric '{metric.value}'.")
                if len(inputs.ground_truth_answers) != len(inputs.queries):
                    raise ValueError("Length of ground truth answers should match the number of queries.")

                eval_inputs[metric.value] = {"ground_truth_answers": inputs.ground_truth_answers}

        return eval_inputs

    @staticmethod
    def _validate_rag_components(
        pipeline: Pipeline,
        components: Dict[RAGExpectedComponent, RAGExpectedComponentMetadata],
        metrics: Set[RAGEvaluationMetric],
    ):
        metric_specific_required_components = {
            RAGEvaluationMetric.DOCUMENT_MAP: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.DOCUMENT_RETRIEVER,
            ],
            RAGEvaluationMetric.DOCUMENT_MRR: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.DOCUMENT_RETRIEVER,
            ],
            RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.DOCUMENT_RETRIEVER,
            ],
            RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.DOCUMENT_RETRIEVER,
            ],
            RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.RESPONSE_GENERATOR,
            ],
            RAGEvaluationMetric.FAITHFULNESS: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.DOCUMENT_RETRIEVER,
                RAGExpectedComponent.RESPONSE_GENERATOR,
            ],
            RAGEvaluationMetric.CONTEXT_RELEVANCE: [
                RAGExpectedComponent.QUERY_PROCESSOR,
                RAGExpectedComponent.DOCUMENT_RETRIEVER,
            ],
        }

        for m in metrics:
            required_components = metric_specific_required_components[m]
            if not all(c in components for c in required_components):
                raise ValueError(
                    f"In order to use the metric '{m}', the RAG evaluation harness requires metadata "
                    f"for the following components: {required_components}"
                )

        pipeline_outputs = pipeline.outputs(include_components_with_connected_outputs=True)
        pipeline_inputs = pipeline.inputs(include_components_with_connected_inputs=True)

        for component, metadata in components.items():
            if metadata.name not in pipeline_outputs or metadata.name not in pipeline_inputs:
                raise ValueError(
                    f"Expected '{component.value}' component named '{metadata.name}' not found in pipeline."
                )

            comp_inputs = pipeline_inputs[metadata.name]
            comp_outputs = pipeline_outputs[metadata.name]

            for needle in metadata.input_mapping.values():
                if needle not in comp_inputs:
                    raise ValueError(
                        f"Required input '{needle}' not found in '{component.value}' "
                        f"component named '{metadata.name}'."
                    )

            for needle in metadata.output_mapping.values():
                if needle not in comp_outputs:
                    raise ValueError(
                        f"Required output '{needle}' not found in '{component.value}' "
                        f"component named '{metadata.name}'."
                    )
