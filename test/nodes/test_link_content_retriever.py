from unittest.mock import Mock, patch
import logging
import pytest
import requests
from requests import Response

from haystack import Document
from haystack.nodes import LinkContentRetriever


@pytest.fixture
def mocked_requests():
    with patch("haystack.nodes.retriever.link_content.requests") as mock_requests:
        mock_response = Mock()
        mock_requests.get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.text = "Sample content from webpage"
        yield mock_requests


@pytest.fixture
def mocked_article_extractor():
    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value="Sample content from webpage"):
        yield


@pytest.mark.unit
def test_init():
    """
    Checks the initialization of the LinkContentRetriever without a preprocessor.
    """
    r = LinkContentRetriever()

    assert r.processor is None
    assert isinstance(r.handlers, dict)
    assert "html" in r.handlers


@pytest.mark.unit
def test_init_with_preprocessor():
    """
    Checks the initialization of the LinkContentRetriever with a preprocessor.
    """
    pre_processor_mock = Mock()
    r = LinkContentRetriever(processor=pre_processor_mock)
    assert r.processor == pre_processor_mock
    assert isinstance(r.handlers, dict)
    assert "html" in r.handlers


@pytest.mark.unit
def test_fetch(mocked_requests, mocked_article_extractor):
    """
    Checks if the LinkContentRetriever is able to fetch content.
    """
    url = "https://haystack.deepset.ai/"

    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    r = LinkContentRetriever(pre_processor_mock)
    result = r.fetch(url=url, doc_kwargs={"text": "Sample content from webpage"})

    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_fetch_no_url(mocked_requests, mocked_article_extractor):
    """
    Ensures an InvalidURL exception is raised when URL is missing.
    """
    pre_processor_mock = Mock()
    pre_processor_mock.process.return_value = [Document("Sample content from webpage")]

    retriever_no_url = LinkContentRetriever(processor=pre_processor_mock)
    with pytest.raises(requests.exceptions.InvalidURL, match="Invalid or missing URL"):
        retriever_no_url.fetch(url="")


@pytest.mark.unit
def test_fetch_invalid_url(caplog, mocked_requests, mocked_article_extractor):
    """
    Ensures an InvalidURL exception is raised when the URL is invalid.
    """
    url = "this-is-invalid-url"

    r = LinkContentRetriever()
    with pytest.raises(requests.exceptions.InvalidURL):
        r.fetch(url=url)


@pytest.mark.unit
def test_fetch_no_preprocessor(mocked_requests, mocked_article_extractor):
    """
    Checks if the LinkContentRetriever can fetch content without a preprocessor.
    """
    url = "https://www.example.com"
    r = LinkContentRetriever()

    result_no_preprocessor = r.fetch(url=url)

    assert len(result_no_preprocessor) == 1
    assert isinstance(result_no_preprocessor[0], Document)
    assert result_no_preprocessor[0].content == "Sample content from webpage"


@pytest.mark.unit
def test_fetch_correct_arguments(mocked_requests, mocked_article_extractor):
    """
    Ensures that requests.get is called with correct arguments.
    """
    url = "https://www.example.com"

    r = LinkContentRetriever()
    r.fetch(url=url)

    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 3
    assert kwargs["headers"] == r.REQUEST_HEADERS

    # another variant
    url = "https://deepset.ai"
    r.fetch(url=url, timeout=10)
    # Check the arguments that requests.get was called with
    args, kwargs = mocked_requests.get.call_args
    assert args[0] == url
    assert kwargs["timeout"] == 10
    assert kwargs["headers"] == r.REQUEST_HEADERS


@pytest.mark.unit
def test_fetch_default_empty_content(mocked_requests):
    """
    Checks handling of content extraction returning empty content.
    """
    url = "https://www.example.com"
    timeout = 10
    content_text = ""
    r = LinkContentRetriever()

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", return_value=content_text):
        result = r.fetch(url=url, timeout=timeout)

    assert "text" not in result
    assert isinstance(result, list) and len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].content == content_text


@pytest.mark.unit
def test_fetch_exception_during_content_extraction_no_raise_on_failure(caplog, mocked_requests):
    """
    Checks the behavior when there's an exception during content extraction, and raise_on_failure is set to False.
    """
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"
    r = LinkContentRetriever()

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", side_effect=Exception("Could not extract content")):
        result = r.fetch(url=url)

    assert "text" not in result
    assert "Couldn't extract content from" in caplog.text


@pytest.mark.unit
def test_fetch_exception_during_content_extraction_raise_on_failure(caplog, mocked_requests):
    """
    Checks the behavior when there's an exception during content extraction, and raise_on_failure is set to True.
    """
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"
    r = LinkContentRetriever(raise_on_failure=True)

    with patch("boilerpy3.extractors.ArticleExtractor.get_content", side_effect=Exception("Could not extract content")):
        with pytest.raises(Exception, match="Could not extract content"):
            r.fetch(url=url)


@pytest.mark.unit
def test_fetch_exception_during_request_get_no_raise_on_failure(caplog):
    """
    Checks the behavior when there's an exception during request.get, and raise_on_failure is set to False.
    """
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"
    r = LinkContentRetriever()

    with patch("haystack.nodes.retriever.link_content.requests.get", side_effect=requests.RequestException()):
        r.fetch(url=url)
    assert f"Couldn't retrieve content from {url}" in caplog.text


@pytest.mark.unit
def test_fetch_exception_during_request_get_raise_on_failure(caplog):
    """
    Checks the behavior when there's an exception during request.get, and raise_on_failure is set to True.
    """
    caplog.set_level(logging.DEBUG)
    url = "https://www.example.com"
    r = LinkContentRetriever(raise_on_failure=True)

    with patch("haystack.nodes.retriever.link_content.requests.get", side_effect=requests.RequestException()):
        with pytest.raises(requests.RequestException):
            r.fetch(url=url)


@pytest.mark.unit
@pytest.mark.parametrize("error_code", [403, 404, 500])
def test_handle_various_response_errors(caplog, mocked_requests, error_code: int):
    """
    Tests the handling of various HTTP error responses.
    """
    caplog.set_level(logging.DEBUG)

    url = "https://some-problematic-url.com"

    # we don't throw exceptions, there might be many of them
    # we log them on debug level
    mock_response = Response()
    mock_response.status_code = error_code
    mocked_requests.get.return_value = mock_response

    r = LinkContentRetriever()
    docs = r.fetch(url=url)

    assert f"Couldn't retrieve content from {url}" in caplog.text
    assert docs == []


@pytest.mark.unit
def test_is_valid_url():
    """
    Checks the _is_valid_url function with a set of valid URLs.
    """
    retriever = LinkContentRetriever()

    valid_urls = [
        "http://www.google.com",
        "https://www.google.com",
        "http://google.com",
        "https://google.com",
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://[::1]",
        "https://[::1]",
        "http://example.com/path/to/page?name=value",
        "https://example.com/path/to/page?name=value",
        "http://example.com:8000",
        "https://example.com:8000",
    ]

    for url in valid_urls:
        assert retriever._is_valid_url(url), f"Expected {url} to be valid"


@pytest.mark.unit
def test_is_invalid_url():
    """
    Checks the _is_valid_url function with a set of invalid URLs.
    """
    retriever = LinkContentRetriever()

    invalid_urls = [
        "http://",
        "https://",
        "http:",
        "https:",
        "www.google.com",
        "google.com",
        "localhost",
        "127.0.0.1",
        "[::1]",
        "/path/to/page",
        "/path/to/page?name=value",
        ":8000",
        "example.com",
        "http:///example.com",
        "https:///example.com",
        "",
        None,
    ]

    for url in invalid_urls:
        assert not retriever._is_valid_url(url), f"Expected {url} to be invalid"


@pytest.mark.integration
def test_call_with_valid_url_on_live_web():
    """
    Test that LinkContentRetriever can fetch content from a valid URL
    """

    retriever = LinkContentRetriever()
    docs = retriever.fetch(url="https://docs.haystack.deepset.ai/", timeout=2)

    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Haystack" in docs[0].content


@pytest.mark.integration
def test_retrieve_with_valid_url_on_live_web():
    """
    Test that LinkContentRetriever can fetch content from a valid URL using the run method
    """

    retriever = LinkContentRetriever()
    docs, _ = retriever.run(query="https://docs.haystack.deepset.ai/")
    docs = docs["documents"]

    assert len(docs) >= 1
    assert isinstance(docs[0], Document)
    assert "Haystack" in docs[0].content


@pytest.mark.integration
def test_retrieve_with_invalid_url():
    """
    Test that LinkContentRetriever raises ValueError when trying to fetch content from an invalid URL
    """

    retriever = LinkContentRetriever()
    with pytest.raises(ValueError):
        retriever.run(query="")


@pytest.mark.integration
def test_retrieve_batch():
    """
    Test that LinkContentRetriever can fetch content from a valid URL using the retrieve_batch method
    """

    retriever = LinkContentRetriever()
    docs, _ = retriever.run_batch(queries=["https://docs.haystack.deepset.ai/", "https://deepset.ai/"])
    assert docs

    docs = docs["documents"]
    # no processor is applied, so each query should return a list of documents with one entry
    assert len(docs) == 2 and len(docs[0]) == 1 and len(docs[1]) == 1

    # each query should return a list of documents
    assert isinstance(docs[0], list) and isinstance(docs[0][0], Document)
    assert isinstance(docs[1], list) and isinstance(docs[1][0], Document)
