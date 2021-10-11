import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rest_api.application import app


@pytest.fixture
def client() -> TestClient:
    os.environ["PIPELINE_YAML_PATH"] = str((Path(__file__).parent / "samples"/"pipeline"/"test_pipeline.yaml").absolute())
    os.environ["QUERY_PIPELINE_NAME"] = "query_pipeline"
    os.environ["INDEXING_PIPELINE_NAME"] = "indexing_pipeline"
    return TestClient(app)

@pytest.fixture
def populated_client(client: TestClient) -> TestClient:
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')
    files_to_upload = [
        {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')},
        {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_2.pdf").open('rb')}
    ]
    for index, fi in enumerate(files_to_upload):
        response = client.post(url="/file-upload", files=fi, data={"meta": f'{{"meta_key": "meta_value", "meta_index": "{index}"}}'})
        assert 200 == response.status_code
    yield client
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')


def test_get_documents(populated_client: TestClient):
    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {"meta_key": ["meta_value"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 2
    names = [doc["meta"]["name"] for doc in response_json]
    assert "sample_pdf_1.pdf" in names
    assert "sample_pdf_2.pdf" in names
    meta_keys = [doc["meta"]["meta_key"] for doc in response_json]
    assert all("meta_value"==meta_key for meta_key in meta_keys)


def test_delete_documents(populated_client: TestClient):
    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {"meta_key": ["meta_value"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 2
    
    response = populated_client.post(url="/documents/delete_by_filters", data='{"filters": {"meta_index": ["0"]}}')
    assert 200 == response.status_code

    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {"meta_key": ["meta_value"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 1
    
    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {"meta_index": ["0"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 0

    response = populated_client.post(url="/documents/get_by_filters", data='{"filters": {"meta_index": ["1"]}}')
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json) == 1

def test_file_upload(client: TestClient):
    file_to_upload = {'files': (Path(__file__).parent / "samples"/"pdf"/"sample_pdf_1.pdf").open('rb')}
    response = client.post(url="/file-upload", files=file_to_upload, data={"meta": '{"meta_key": "meta_value"}'})
    assert 200 == response.status_code
    client.post(url="/documents/delete_by_filters", data='{"filters": {}}')

def test_query_with_no_filter(populated_client: TestClient):
    query_with_no_filter_value = {"query": "Who made the PDF specification?"}
    response = populated_client.post(url="/query", json=query_with_no_filter_value)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

def test_query_with_one_filter(populated_client: TestClient):
    query_with_filter = {"query": "Who made the PDF specification?", "params": {"filters": {"meta_key": "meta_value"}}}
    response = populated_client.post(url="/query", json=query_with_filter)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

def test_query_with_filter_list(populated_client: TestClient):
    query_with_filter_list = {
        "query": "Who made the PDF specification?",
        "params": {"filters": {"meta_key": ["meta_value", "another_value"]}}
    }
    response = populated_client.post(url="/query", json=query_with_filter_list)
    assert 200 == response.status_code
    response_json = response.json()
    assert response_json["answers"][0]["answer"] == "Adobe Systems"

def test_query_with_invalid_filter(populated_client: TestClient):
    query_with_invalid_filter = {
        "query": "Who made the PDF specification?", "params": {"filters": {"meta_key": "invalid_value"}}
    }
    response = populated_client.post(url="/query", json=query_with_invalid_filter)
    assert 200 == response.status_code
    response_json = response.json()
    assert len(response_json["answers"]) == 0

def test_write_feedback(populated_client: TestClient):
    response = populated_client.post(url="/query", json={"query": "Who made the PDF specification?"})
    response_json = response.json()
    document_id = response_json["answers"][0]["document_id"]

    feedback = {
        "question": "Who made the PDF specification?",
        "is_correct_answer": True,
        "document_id": document_id,
        "is_correct_document": True,
        "answer": "Adobe Systems",
        "offset_start_in_doc": 60
    }
    response = populated_client.post(url="/feedback", json=feedback)
    assert 200 == response.status_code

def test_export_feedback(populated_client: TestClient):
    response = populated_client.post(url="/query", json={"query": "Who made the PDF specification?"})
    response_json = response.json()
    document_id = response_json["answers"][0]["document_id"]

    feedback = {
        "question": "Who made the PDF specification?",
        "is_correct_answer": True,
        "document_id": document_id,
        "is_correct_document": True,
        "answer": "Adobe Systems",
        "offset_start_in_doc": 60
    }
    feedback_urls = [
        "/export-feedback?full_document_context=true",
        "/export-feedback?full_document_context=false&context_size=50",
        "/export-feedback?full_document_context=false&context_size=50000",
    ]
    for url in feedback_urls:
        response = populated_client.get(url=url, json=feedback)
        response_json = response.json()
        context = response_json["data"][0]["paragraphs"][0]["context"]
        answer_start = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"]
        answer = response_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
        assert context[answer_start:answer_start+len(answer)] == answer
