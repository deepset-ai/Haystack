# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union
import pytest

from haystack.evaluation.harness.rag import (
    DefaultRAGArchitecture,
    RAGEvaluationHarness,
    RAGExpectedComponent,
    RAGExpectedComponentMetadata,
    RAGEvaluationMetric,
    RAGEvaluationOverrides,
    RAGEvaluationInput,
)
from haystack import Pipeline, component, Document, default_to_dict, default_from_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
    FaithfulnessEvaluator,
    SASEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret


@component
class NonConformantComponent:
    def __init__(self, inputs, outputs) -> None:
        component.set_input_types(self, **inputs)
        component.set_output_types(self, **outputs)

    def run(self, **kwargs):
        return {}


@component
class MockGenerator:
    def __init__(self, arg: int) -> None:
        self.arg = arg

    def to_dict(self):
        return default_to_dict(self, arg=self.arg)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str])
    def run(self, prompt: str) -> Dict[str, Any]:
        return {"replies": ["placeholder"]}


@component
class MockKeywordRetriever:
    def __init__(self) -> None:
        self.counter = 0

    @component.output_types(documents=List[Document])
    def run(self, query: str) -> Dict[str, Any]:
        samples = [
            [Document(content="France")],
            [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
            [Document(content="classical"), Document(content="rock music"), Document(content="dubstep")],
            [Document(content="11th"), Document(content="the 11th"), Document(content="11th century")],
            [Document(content="Denmark"), Document(content="Norway"), Document(content="Iceland")],
            [
                Document(content="10th century"),
                Document(content="the first half of the 10th century"),
                Document(content="10th"),
                Document(content="10th"),
            ],
        ]

        idx = self.counter % len(samples)
        self.counter += 1

        return {"documents": samples[idx]}


@component
class MockEvaluator:
    def __init__(self, metric: Union[str, RAGEvaluationMetric]) -> None:
        if isinstance(metric, str):
            metric = RAGEvaluationMetric(metric)
        self.metric = metric

        io_map = {
            RAGEvaluationMetric.DOCUMENT_MAP: DocumentMAPEvaluator(),
            RAGEvaluationMetric.DOCUMENT_MRR: DocumentMRREvaluator(),
            RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT: DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT),
            RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT: DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT),
            RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY: SASEvaluator("sentence-transformers/all-MiniLM-L6-v2"),
            RAGEvaluationMetric.FAITHFULNESS: FaithfulnessEvaluator(api_key=Secret.from_token("test_key")),
            RAGEvaluationMetric.CONTEXT_RELEVANCE: ContextRelevanceEvaluator(api_key=Secret.from_token("test_key")),
        }

        self.__haystack_input__ = io_map[metric].__haystack_input__
        self.__haystack_output__ = io_map[metric].__haystack_output__

    def to_dict(self):
        return default_to_dict(self, metric=str(self.metric))

    @staticmethod
    def default_output(metric) -> Dict[str, Any]:
        if metric in (
            RAGEvaluationMetric.FAITHFULNESS,
            RAGEvaluationMetric.DOCUMENT_MAP,
            RAGEvaluationMetric.DOCUMENT_MRR,
            RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT,
            RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT,
        ):
            return {"individual_scores": [1] * 6, "score": 1.0}
        else:
            return {
                "individual_scores": [1] * 6,
                "score": 1.0,
                "results": [{"statements": ["placeholder"], "statement_scores": [1.0], "score": 1.0}] * 6,
            }

    def run(self, **kwargs) -> Dict[str, Any]:
        return self.default_output(self.metric)


def build_rag_pipeline_with_query_embedder(
    embedder_name: str = "text_embedder",
    embedder_component: Optional[Any] = None,
    generator_name: str = "llm",
    generator_component: Optional[Any] = None,
):
    document_store = InMemoryDocumentStore()
    retriever = InMemoryEmbeddingRetriever(document_store)

    if embedder_component:
        text_embedder = embedder_component
    else:
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)

    if generator_component:
        generator = generator_component
    else:
        generator = OpenAIGenerator(model="gpt-3.5-turbo", api_key=Secret.from_token("test_key"))

    pipeline = Pipeline()
    pipeline.add_component(embedder_name, text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component(generator_name, generator)
    pipeline.connect(f"{embedder_name}.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", generator_name)
    return pipeline


def build_rag_pipeline_with_keyword_retriever(
    retriever_name: str = "retriever",
    retriever_component: Optional[Any] = None,
    retriever_output_name: str = "documents",
    generator_name: str = "llm",
    generator_component: Optional[Any] = None,
):
    document_store = InMemoryDocumentStore()
    if retriever_component:
        retriever = retriever_component
    else:
        retriever = InMemoryBM25Retriever(document_store)
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)
    if generator_component:
        generator = generator_component
    else:
        generator = OpenAIGenerator(model="gpt-3.5-turbo", api_key=Secret.from_token("test_key"))

    pipeline = Pipeline()
    pipeline.add_component(retriever_name, retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component(generator_name, generator)
    pipeline.connect(f"{retriever_name}.{retriever_output_name}", "prompt_builder.documents")
    pipeline.connect("prompt_builder", generator_name)
    return pipeline


@pytest.fixture
def rag_pipeline():
    return build_rag_pipeline_with_query_embedder("text_embedder")


@pytest.fixture
def rag_pipeline_with_query_embedder():
    return build_rag_pipeline_with_query_embedder(embedder_name="query_embedder", generator_name="generator")


@pytest.fixture
def rag_pipeline_with_keyword_retriever():
    return build_rag_pipeline_with_keyword_retriever(generator_name="generator")


class TestRAGEvaluationHarness:
    def test_init(self, rag_pipeline):
        harness = RAGEvaluationHarness(
            rag_pipeline,
            rag_components={
                RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                    name="text_embedder", input_mapping={"query": "text"}
                ),
                RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                    name="retriever", output_mapping={"retrieved_documents": "documents"}
                ),
                RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
                    name="llm", output_mapping={"replies": "replies"}
                ),
            },
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

    def test_init_invalid_expected_component(self, rag_pipeline):
        with pytest.raises(ValueError, match="RAG evaluation harness requires metadata"):
            _ = RAGEvaluationHarness(rag_pipeline, rag_components={}, metrics={RAGEvaluationMetric.DOCUMENT_MAP})

        with pytest.raises(ValueError, match="RAG evaluation harness requires metadata"):
            _ = RAGEvaluationHarness(
                rag_pipeline,
                rag_components={
                    RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                        name="text_embedder", input_mapping={"query": "text"}
                    )
                },
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

    def test_init_invalid_missing_components(self, rag_pipeline):
        with pytest.raises(ValueError, match="named 'embedder' not found in pipeline"):
            _ = RAGEvaluationHarness(
                rag_pipeline,
                rag_components={
                    RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                        name="embedder", input_mapping={"query": "text"}
                    ),
                    RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                        name="retriever", output_mapping={"retrieved_documents": "documents"}
                    ),
                    RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
                        name="llm", output_mapping={"replies": "replies"}
                    ),
                },
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

    def test_init_invalid_missing_inputs(self, rag_pipeline):
        with pytest.raises(
            ValueError,
            match="Required input 'rando_input' not found in 'query_processor' component named 'text_embedder'",
        ):
            _ = RAGEvaluationHarness(
                rag_pipeline,
                rag_components={
                    RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                        name="text_embedder", input_mapping={"query": "rando_input"}
                    ),
                    RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                        name="retriever", output_mapping={"retrieved_documents": "documents"}
                    ),
                    RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
                        name="llm", output_mapping={"replies": "replies"}
                    ),
                },
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

    def test_init_invalid_missing_outputs(self, rag_pipeline):
        with pytest.raises(
            ValueError, match="Required output 'rando_output' not found in 'response_generator' component named 'llm'"
        ):
            _ = RAGEvaluationHarness(
                rag_pipeline,
                rag_components={
                    RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                        name="text_embedder", input_mapping={"query": "text"}
                    ),
                    RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                        name="retriever", output_mapping={"retrieved_documents": "documents"}
                    ),
                    RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
                        name="llm", output_mapping={"replies": "rando_output"}
                    ),
                },
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

    def test_init_defaults(self, rag_pipeline_with_query_embedder, rag_pipeline_with_keyword_retriever):
        _ = RAGEvaluationHarness(
            rag_pipeline_with_query_embedder,
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

        _ = RAGEvaluationHarness(
            rag_pipeline_with_keyword_retriever,
            DefaultRAGArchitecture.GENERATION_WITH_KEYWORD_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

        _ = RAGEvaluationHarness(
            rag_pipeline_with_query_embedder,
            DefaultRAGArchitecture.EMBEDDING_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

        _ = RAGEvaluationHarness(
            rag_pipeline_with_keyword_retriever,
            DefaultRAGArchitecture.KEYWORD_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

    def test_init_defaults_invalid_missing_inputs(self):
        with pytest.raises(
            ValueError, match="Required input 'text' not found in 'query_processor' component named 'query_embedder'"
        ):
            _ = RAGEvaluationHarness(
                build_rag_pipeline_with_query_embedder(embedder_name="llm", generator_name="query_embedder"),
                DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

        with pytest.raises(
            ValueError, match="Required input 'query' not found in 'query_processor' component named 'retriever'"
        ):
            _ = RAGEvaluationHarness(
                build_rag_pipeline_with_keyword_retriever(retriever_name="llm", generator_name="retriever"),
                DefaultRAGArchitecture.GENERATION_WITH_KEYWORD_RETRIEVAL,
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

    def test_init_defaults_invalid_missing_outputs(self):
        non_conformant_query_embedder_pipeline = build_rag_pipeline_with_query_embedder(
            embedder_name="query_embedder",
            generator_name="generator",
            generator_component=NonConformantComponent({"prompt": str}, {"responses": List[str]}),
        )
        non_conformant_keyword_retriever_pipeline = build_rag_pipeline_with_keyword_retriever(
            retriever_component=NonConformantComponent({"query": str}, {"docs": List[Document]}),
            retriever_output_name="docs",
        )

        with pytest.raises(
            ValueError, match="Required output 'replies' not found in 'response_generator' component named 'generator'"
        ):
            _ = RAGEvaluationHarness(
                non_conformant_query_embedder_pipeline,
                DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

        with pytest.raises(
            ValueError,
            match="Required output 'documents' not found in 'document_retriever' component named 'retriever'",
        ):
            _ = RAGEvaluationHarness(
                non_conformant_keyword_retriever_pipeline,
                DefaultRAGArchitecture.GENERATION_WITH_KEYWORD_RETRIEVAL,
                metrics={RAGEvaluationMetric.DOCUMENT_MAP},
            )

    def test_init_invalid_component_for_metric(self, rag_pipeline_with_query_embedder):
        with pytest.raises(ValueError, match="In order to use the metric .* RAG evaluation harness requires metadata"):
            _ = RAGEvaluationHarness(
                rag_pipeline_with_query_embedder,
                DefaultRAGArchitecture.EMBEDDING_RETRIEVAL,
                metrics={RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY},
            )

    def test_run_invalid_ground_truths(self, rag_pipeline_with_query_embedder):
        harness_map = RAGEvaluationHarness(
            rag_pipeline_with_query_embedder,
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )
        harness_sas = RAGEvaluationHarness(
            rag_pipeline_with_query_embedder,
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
            metrics={RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY},
        )

        input_no_gt_docs = RAGEvaluationInput(queries=["What is the capital of France?"])
        input_mismatching_gt_docs = RAGEvaluationInput(
            queries=["What is the capital of France?"], ground_truth_documents=[]
        )
        input_no_gt_answers = RAGEvaluationInput(
            queries=["What is the capital of France?"],
            ground_truth_documents=[[Document(content="Paris is the capital of France.")]],
        )
        input_mismatching_gt_answers = RAGEvaluationInput(
            queries=["What is the capital of France?"],
            ground_truth_documents=[[Document(content="Paris is the capital of France.")]],
            ground_truth_answers=[],
        )

        with pytest.raises(ValueError, match="Ground truth documents required"):
            _ = harness_map.run(input_no_gt_docs)

        with pytest.raises(ValueError, match="Length of ground truth documents should match the number of queries"):
            _ = harness_map.run(input_mismatching_gt_docs)

        with pytest.raises(ValueError, match="Ground truth answers required"):
            _ = harness_sas.run(input_no_gt_answers)

        with pytest.raises(ValueError, match="Length of ground truth answers should match the number of queries"):
            _ = harness_sas.run(input_mismatching_gt_answers)

    def test_run_invalid_additional_input(self, rag_pipeline_with_query_embedder):
        harness = RAGEvaluationHarness(
            rag_pipeline_with_query_embedder,
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

        input = RAGEvaluationInput(
            queries=["What is the capital of France?"],
            ground_truth_documents=[[Document(content="Paris is the capital of France.")]],
            rag_pipeline_inputs={"query_embedder": {"text": ["Some other question?"]}},
        )

        with pytest.raises(ValueError, match="Query embedder input 'text' cannot be provided as additional input"):
            _ = harness.run(input)

    def test_run_invalid_override(self, rag_pipeline_with_query_embedder):
        harness = RAGEvaluationHarness(
            rag_pipeline_with_query_embedder,
            DefaultRAGArchitecture.GENERATION_WITH_EMBEDDING_RETRIEVAL,
            metrics={RAGEvaluationMetric.DOCUMENT_MAP},
        )

        input = RAGEvaluationInput(
            queries=["What is the capital of France?"],
            ground_truth_documents=[[Document(content="Paris is the capital of France.")]],
        )

        with pytest.raises(ValueError, match="Cannot override non-existent component 'rando_component'"):
            _ = harness.run(
                input, overrides=RAGEvaluationOverrides(rag_pipeline={"rando_component": {"Some": "thing"}})
            )

        with pytest.raises(ValueError, match="Cannot override parameters of unused evaluation metric"):
            _ = harness.run(
                input,
                overrides=RAGEvaluationOverrides(
                    eval_pipeline={RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT: {"mode": "single_hit"}}
                ),
            )

    def test_run_statistical_metrics(self):
        metrics = {
            RAGEvaluationMetric.DOCUMENT_MAP,
            RAGEvaluationMetric.DOCUMENT_MRR,
            RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT,
            RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT,
        }
        harness = RAGEvaluationHarness(
            build_rag_pipeline_with_keyword_retriever(
                retriever_component=MockKeywordRetriever(),
                generator_component=MockGenerator(arg=0),
                generator_name="generator",
            ),
            DefaultRAGArchitecture.KEYWORD_RETRIEVAL,
            metrics=metrics,
        )

        mock_eval_pipeline = Pipeline()
        for m in metrics:
            mock_eval_pipeline.add_component(m.value, MockEvaluator(metric=m))

        harness.evaluation_pipeline = mock_eval_pipeline

        inputs = RAGEvaluationInput(
            queries=["What is the capital of France?"] * 6,
            ground_truth_documents=[
                [Document(content="France")],
                [Document(content="9th century"), Document(content="9th")],
                [Document(content="classical music"), Document(content="classical")],
                [Document(content="11th century"), Document(content="the 11th")],
                [Document(content="Denmark, Iceland and Norway")],
                [Document(content="10th century"), Document(content="10th")],
            ],
        )

        output = harness.run(
            inputs, overrides=RAGEvaluationOverrides(rag_pipeline={"generator": {"arg": 100}}), run_name="test_run"
        )

        assert output.inputs == inputs
        assert output.results.run_name == "test_run"
        assert output.results.results == {m.value: MockEvaluator.default_output(m) for m in metrics}
        overriden_pipeline_dict = Pipeline.loads(output.evaluated_pipeline).to_dict()
        assert overriden_pipeline_dict["components"]["generator"]["init_parameters"]["arg"] == 100

    def test_run_model_based_metrics(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")

        metrics = {
            RAGEvaluationMetric.FAITHFULNESS,
            RAGEvaluationMetric.CONTEXT_RELEVANCE,
            RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY,
        }
        harness = RAGEvaluationHarness(
            build_rag_pipeline_with_keyword_retriever(
                retriever_component=MockKeywordRetriever(),
                generator_component=MockGenerator(arg=0),
                generator_name="generator",
            ),
            DefaultRAGArchitecture.GENERATION_WITH_KEYWORD_RETRIEVAL,
            metrics=metrics,
        )

        mock_eval_pipeline = Pipeline()
        for m in metrics:
            mock_eval_pipeline.add_component(m.value, MockEvaluator(metric=m))

        harness.evaluation_pipeline = mock_eval_pipeline

        inputs = RAGEvaluationInput(
            queries=["What is the capital of France?"] * 6,
            ground_truth_documents=[
                [Document(content="France")],
                [Document(content="9th century"), Document(content="9th")],
                [Document(content="classical music"), Document(content="classical")],
                [Document(content="11th century"), Document(content="the 11th")],
                [Document(content="Denmark, Iceland and Norway")],
                [Document(content="10th century"), Document(content="10th")],
            ],
            ground_truth_answers=[
                "Paris is the capital of France.",
                "9th century",
                "classical music",
                "11th century",
                "Denmark, Iceland and Norway",
                "10th century",
            ],
        )

        output = harness.run(inputs, run_name="test_run")

        assert output.inputs == inputs
        assert output.results.run_name == "test_run"
        assert output.results.inputs == {
            "questions": ["What is the capital of France?"] * 6,
            "contexts": [
                ["France"],
                ["9th century", "10th century", "9th"],
                ["classical", "rock music", "dubstep"],
                ["11th", "the 11th", "11th century"],
                ["Denmark", "Norway", "Iceland"],
                ["10th century", "the first half of the 10th century", "10th", "10th"],
            ],
            "responses": ["placeholder", "placeholder", "placeholder", "placeholder", "placeholder", "placeholder"],
            "ground_truth_documents": [
                ["France"],
                ["9th century", "9th"],
                ["classical music", "classical"],
                ["11th century", "the 11th"],
                ["Denmark, Iceland and Norway"],
                ["10th century", "10th"],
            ],
            "ground_truth_answers": [
                "Paris is the capital of France.",
                "9th century",
                "classical music",
                "11th century",
                "Denmark, Iceland and Norway",
                "10th century",
            ],
        }
        assert output.results.results == {m.value: MockEvaluator.default_output(m) for m in metrics}
