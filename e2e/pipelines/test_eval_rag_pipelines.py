from haystack import Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.evaluation.eval import eval


def test_bm25_rag_pipeline():
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()), name="retriever")
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    rag_pipeline.get_component("retriever").document_store.write_documents(documents)

    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
        for question in questions
    ]

    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]

    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == rag_pipeline.to_dict()


def test_embedding_retrieval_rag_pipeline():
    # Create the RAG pipeline
    prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(
        instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="text_embedder",
    )
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()), name="retriever"
    )
    rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(
        instance=HuggingFaceLocalGenerator(
            model_name_or_path="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
        ),
        name="llm",
    )
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag_pipeline.connect("text_embedder", "retriever")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    # Populate the document store
    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    document_store = rag_pipeline.get_component("retriever").document_store
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
        name="document_embedder",
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})

    # Query and assert
    questions = ["Who lives in Paris?", "Who lives in Berlin?", "Who lives in Rome?"]
    inputs = [
        {
            "prompt_builder": {"question": question},
            "text_embedder": {"text": question},
            "answer_builder": {"query": question},
        }
        for question in questions
    ]

    expected_outputs = [
        {"llm": {"replies": ["Jean"]}},
        {"llm": {"replies": ["Mark"]}},
        {"llm": {"replies": ["Giorgio"]}},
    ]

    eval_result = eval(rag_pipeline, inputs=inputs, expected_outputs=expected_outputs)

    assert eval_result.inputs == inputs
    assert eval_result.expected_outputs == expected_outputs
    assert len(eval_result.outputs) == len(expected_outputs) == len(inputs)
    assert eval_result.runnable.to_dict() == rag_pipeline.to_dict()
