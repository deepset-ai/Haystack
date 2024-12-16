import pytest
from pytest import LogCaptureFixture

from haystack import Document, Pipeline
from haystack.components.preprocessors.recursive_splitter import RecursiveDocumentSplitter
from haystack.components.preprocessors.sentence_tokenizer import SentenceSplitter


def test_get_custom_sentence_tokenizer_success():
    tokenizer = RecursiveDocumentSplitter._get_custom_sentence_tokenizer()
    assert isinstance(tokenizer, SentenceSplitter)


def test_init_with_negative_overlap():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=20, split_overlap=-1, separators=["."])


def test_init_with_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=10, split_overlap=15, separators=["."])


def test_init_with_invalid_separators():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(separators=[".", 2])


def test_init_with_negative_split_length():
    with pytest.raises(ValueError):
        _ = RecursiveDocumentSplitter(split_length=-1, separators=["."])


def test_apply_overlap_no_overlap():
    # Test the case where there is no overlap between chunks
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "chunk2", "chunk3"]


def test_apply_overlap_with_overlap_case_1():
    # Test the case where there is overlap between chunks
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=4, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "unk1chunk2", "unk2chunk3"]


# ToDo: update this test, result above is not the expected one
def ignore_test_apply_overlap_with_overlap_case_2():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=6, separators=["."])
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1", "chunk1chunk2", "chunk2chunk3", "chunk3chunk4"]


def test_apply_overlap_single_chunk():
    # Test the case where there is only one chunk
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=3, separators=["."])
    chunks = ["chunk1"]
    result = splitter._apply_overlap(chunks)
    assert result == ["chunk1"]


def test_chunk_text_smaller_than_chunk_size():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    text = "small text"
    chunks = splitter._chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_by_period():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    text = "This is a test. Another sentence. And one more."
    chunks = splitter._chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "This is a test."
    assert chunks[1] == " Another sentence."
    assert chunks[2] == " And one more."


def test_recursive_splitter_multiple_new_lines():
    splitter = RecursiveDocumentSplitter(split_length=20, separators=["\n\n", "\n"])
    text = "This is a test.\n\n\nAnother test.\n\n\n\nFinal test."
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    assert chunks[0].content == "This is a test.\n\n"
    assert chunks[1].content == "\nAnother test.\n\n"
    assert chunks[2].content == "\n\nFinal test."


def test_recursive_splitter_empty_documents(caplog: LogCaptureFixture):
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=0, separators=["."])
    empty_doc = Document(content="")
    doc_chunks = splitter.run([empty_doc])
    doc_chunks = doc_chunks["documents"]
    assert len(doc_chunks) == 0
    assert "has an empty content. Skipping this document." in caplog.text


def test_recursive_splitter_using_custom_sentence_tokenizer():
    """
    This test includes abbreviations that are not handled by the simple sentence tokenizer based on "." and requires a
    more sophisticated sentence tokenizer like the one provided by NLTK.
    """
    splitter = RecursiveDocumentSplitter(split_length=400, split_overlap=0, separators=["\n\n", "\n", "sentence", " "])
    text = """Artificial intelligence (AI) - Introduction

AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.
AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."""  # noqa: E501

    chunks = splitter.run([Document(content=text)])
    chunks = chunks["documents"]
    assert len(chunks) == 4
    assert chunks[0].content == "Artificial intelligence (AI) - Introduction\n\n"
    assert (
        chunks[1].content
        == "AI, in its broadest sense, is intelligence exhibited by machines, particularly computer systems.\n"
    )  # noqa: E501
    assert chunks[2].content == "AI technology is widely used throughout industry, government, and science."  # noqa: E501
    assert (
        chunks[3].content
        == "Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go)."
    )  # noqa: E501


def test_run_split_by_dot_count_page_breaks() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["."], split_length=30, split_overlap=0)

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])["documents"]

    assert len(documents) == 7
    assert documents[0].content == "Sentence on page 1."
    assert documents[0].meta["page_number"] == 1
    assert documents[0].meta["split_id"] == 0
    assert documents[0].meta["split_idx_start"] == text.index(documents[0].content)

    assert documents[1].content == " Another on page 1."
    assert documents[1].meta["page_number"] == 1
    assert documents[1].meta["split_id"] == 1
    assert documents[1].meta["split_idx_start"] == text.index(documents[1].content)

    assert documents[2].content == "\fSentence on page 2."
    assert documents[2].meta["page_number"] == 2
    assert documents[2].meta["split_id"] == 2
    assert documents[2].meta["split_idx_start"] == text.index(documents[2].content)

    assert documents[3].content == " Another on page 2."
    assert documents[3].meta["page_number"] == 2
    assert documents[3].meta["split_id"] == 3
    assert documents[3].meta["split_idx_start"] == text.index(documents[3].content)

    assert documents[4].content == "\fSentence on page 3."
    assert documents[4].meta["page_number"] == 3
    assert documents[4].meta["split_id"] == 4
    assert documents[4].meta["split_idx_start"] == text.index(documents[4].content)

    assert documents[5].content == " Another on page 3."
    assert documents[5].meta["page_number"] == 3
    assert documents[5].meta["split_id"] == 5
    assert documents[5].meta["split_idx_start"] == text.index(documents[5].content)

    assert documents[6].content == "\f\f Sentence on page 5."
    assert documents[6].meta["page_number"] == 5
    assert documents[6].meta["split_id"] == 6
    assert documents[6].meta["split_idx_start"] == text.index(documents[6].content)


def test_run_split_by_word_count_page_breaks():
    splitter = RecursiveDocumentSplitter(split_length=18, split_overlap=0, separators=["w"])
    text = "This is some text. \f This text is on another page. \f This is the last pag3."
    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 5
    assert doc_chunks[0].content == "This is some text."
    assert doc_chunks[0].meta["page_number"] == 1
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)

    assert doc_chunks[1].content == " \f This text is on"
    assert doc_chunks[1].meta["page_number"] == 2
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)

    assert doc_chunks[2].content == " another page. \f T"
    assert doc_chunks[2].meta["page_number"] == 3
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)

    assert doc_chunks[3].content == "his is the last pa"
    assert doc_chunks[3].meta["page_number"] == 3
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == text.index(doc_chunks[3].content)

    assert doc_chunks[4].content == "g3."
    assert doc_chunks[4].meta["page_number"] == 3
    assert doc_chunks[4].meta["split_id"] == 4
    assert doc_chunks[4].meta["split_idx_start"] == text.index(doc_chunks[4].content)


def test_run_split_by_page_break_count_page_breaks() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["\f"], split_length=50, split_overlap=0)

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]
    assert len(chunks_docs) == 4
    assert chunks_docs[0].content == "Sentence on page 1. Another on page 1.\f"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Sentence on page 2. Another on page 2.\f"
    assert chunks_docs[1].meta["page_number"] == 2
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "Sentence on page 3. Another on page 3.\f\f"
    assert chunks_docs[2].meta["page_number"] == 3
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == " Sentence on page 5."
    assert chunks_docs[3].meta["page_number"] == 5
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)


def test_run_split_by_new_line_count_page_breaks() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["\n"], split_length=50, split_overlap=0)

    text = (
        "Sentence on page 1.\nAnother on page 1.\f"
        "Sentence on page 2.\nAnother on page 2.\f"
        "Sentence on page 3.\nAnother on page 3.\f\f"
        "Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]
    assert len(chunks_docs) == 7

    assert chunks_docs[0].content == "Sentence on page 1.\n"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Another on page 1.\f"
    assert chunks_docs[1].meta["page_number"] == 1
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    assert chunks_docs[2].content == "Sentence on page 2.\n"
    assert chunks_docs[2].meta["page_number"] == 2
    assert chunks_docs[2].meta["split_id"] == 2
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[2].content)

    assert chunks_docs[3].content == "Another on page 2.\f"
    assert chunks_docs[3].meta["page_number"] == 2
    assert chunks_docs[3].meta["split_id"] == 3
    assert chunks_docs[3].meta["split_idx_start"] == text.index(chunks_docs[3].content)

    assert chunks_docs[4].content == "Sentence on page 3.\n"
    assert chunks_docs[4].meta["page_number"] == 3
    assert chunks_docs[4].meta["split_id"] == 4
    assert chunks_docs[4].meta["split_idx_start"] == text.index(chunks_docs[4].content)

    assert chunks_docs[5].content == "Another on page 3.\f\f"
    assert chunks_docs[5].meta["page_number"] == 3
    assert chunks_docs[5].meta["split_id"] == 5
    assert chunks_docs[5].meta["split_idx_start"] == text.index(chunks_docs[5].content)

    assert chunks_docs[6].content == "Sentence on page 5."
    assert chunks_docs[6].meta["page_number"] == 5
    assert chunks_docs[6].meta["split_id"] == 6
    assert chunks_docs[6].meta["split_idx_start"] == text.index(chunks_docs[6].content)


def test_run_split_by_sentence_count_page_breaks() -> None:
    document_splitter = RecursiveDocumentSplitter(separators=["sentence"], split_length=50, split_overlap=0)

    text = (
        "Sentence on page 1. Another on page 1.\fSentence on page 2. Another on page 2.\f"
        "Sentence on page 3. Another on page 3.\f\f Sentence on page 5."
    )

    documents = document_splitter.run(documents=[Document(content=text)])
    chunks_docs = documents["documents"]
    assert len(chunks_docs) == 5

    print("\n-----------")
    for chunk in chunks_docs:
        print(chunk.content)
        print(chunk.meta)
        print("\n-----------")

    assert chunks_docs[0].content == "Sentence on page 1. Another on page 1.\f"
    assert chunks_docs[0].meta["page_number"] == 1
    assert chunks_docs[0].meta["split_id"] == 0
    assert chunks_docs[0].meta["split_idx_start"] == text.index(chunks_docs[0].content)

    assert chunks_docs[1].content == "Sentence on page 2. "
    assert chunks_docs[1].meta["page_number"] == 2
    assert chunks_docs[1].meta["split_id"] == 1
    assert chunks_docs[1].meta["split_idx_start"] == text.index(chunks_docs[1].content)

    # assert chunks_docs[2].content == "\fSentence on page 3. Another on page 3.\f"
    assert chunks_docs[2].meta["page_number"] == 3
    assert chunks_docs[2].meta["split_id"] == 3
    assert chunks_docs[2].meta["split_idx_start"] == text.index(chunks_docs[4].content)
    #
    # assert chunks_docs[5].content == " Another on page 3."
    # assert chunks_docs[5].meta["page_number"] == 3
    # assert chunks_docs[5].meta["split_id"] == 4
    # assert chunks_docs[5].meta["split_idx_start"] == text.index(chunks_docs[5].content)
    #
    # assert chunks_docs[6].content == "\f\f Sentence on page 5."
    # assert chunks_docs[6].meta["page_number"] == 5
    # assert chunks_docs[6].meta["split_id"] == 5
    # assert chunks_docs[6].meta["split_idx_start"] == text.index(chunks_docs[6].content)


def test_recursive_splitter_custom_sentence_tokenizer_document_and_overlap():
    """Test that RecursiveDocumentSplitter works correctly with custom sentence tokenizer and overlap"""
    splitter = RecursiveDocumentSplitter(split_length=25, split_overlap=5, separators=["sentence"])
    text = "This is sentence one. This is sentence two. This is sentence three."

    doc = Document(content=text)
    doc_chunks = splitter.run([doc])["documents"]

    assert len(doc_chunks) == 3

    assert doc_chunks[0].content == "This is sentence one. "
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)
    assert doc_chunks[0].meta["_split_overlap"] == [{"doc_id": doc_chunks[1].id, "range": (0, 5)}]

    assert doc_chunks[1].content == "one. This is sentence two. "
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)
    assert doc_chunks[1].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[0].id, "range": (17, 22)},
        {"doc_id": doc_chunks[2].id, "range": (0, 5)},
    ]

    assert doc_chunks[2].content == "two. This is sentence three."
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)
    assert doc_chunks[2].meta["_split_overlap"] == [{"doc_id": doc_chunks[1].id, "range": (22, 27)}]


def test_run_split_document_with_overlap():
    splitter = RecursiveDocumentSplitter(split_length=20, split_overlap=11, separators=[".", " "])
    text = """A simple sentence1. A bright sentence2. A clever sentence3. A joyful sentence4"""

    doc = Document(content=text)
    doc_chunks = splitter.run([doc])
    doc_chunks = doc_chunks["documents"]

    assert len(doc_chunks) == 4

    assert doc_chunks[0].content == "A simple sentence1."
    assert doc_chunks[0].meta["split_id"] == 0
    assert doc_chunks[0].meta["split_idx_start"] == text.index(doc_chunks[0].content)
    assert doc_chunks[0].meta["_split_overlap"] == [{"doc_id": doc_chunks[1].id, "range": (0, 11)}]

    assert doc_chunks[1].content == " sentence1. A bright sentence2."
    assert doc_chunks[1].meta["split_id"] == 1
    assert doc_chunks[1].meta["split_idx_start"] == text.index(doc_chunks[1].content)
    assert doc_chunks[1].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[0].id, "range": (8, 19)},
        {"doc_id": doc_chunks[2].id, "range": (0, 11)},
    ]

    assert doc_chunks[2].content == " sentence2. A clever sentence3."
    assert doc_chunks[2].meta["split_id"] == 2
    assert doc_chunks[2].meta["split_idx_start"] == text.index(doc_chunks[2].content)
    assert doc_chunks[2].meta["_split_overlap"] == [
        {"doc_id": doc_chunks[1].id, "range": (20, 31)},
        {"doc_id": doc_chunks[3].id, "range": (0, 11)},
    ]

    assert doc_chunks[3].content == " sentence3. A joyful sentence4"
    assert doc_chunks[3].meta["split_id"] == 3
    assert doc_chunks[3].meta["split_idx_start"] == text.index(doc_chunks[3].content)
    assert doc_chunks[3].meta["_split_overlap"] == [{"doc_id": doc_chunks[2].id, "range": (20, 31)}]


def test_run_separator_exists_but_split_length_too_small_fall_back_to_character_chunking():
    splitter = RecursiveDocumentSplitter(separators=[" "], split_length=2)
    doc = Document(content="This is some text. This is some more text.")
    result = splitter.run(documents=[doc])
    assert len(result["documents"]) == 21
    for doc in result["documents"]:
        assert len(doc.content) == 2


def test_run_fallback_to_character_chunking():
    text = "abczdefzghizjkl"
    separators = ["\n\n", "\n", "z"]
    splitter = RecursiveDocumentSplitter(split_length=2, separators=separators)
    doc = Document(content=text)
    chunks = splitter.run([doc])["documents"]
    for chunk in chunks:
        assert len(chunk.content) <= 2


def test_run_serialization_in_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("chunker", RecursiveDocumentSplitter(split_length=20, split_overlap=5, separators=["."]))
    pipeline_dict = pipeline.dumps()
    new_pipeline = Pipeline.loads(pipeline_dict)
    assert pipeline_dict == new_pipeline.dumps()
