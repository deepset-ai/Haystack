import tarfile
import time
import urllib.request
from subprocess import Popen, PIPE, STDOUT, run
import os

import pytest
from elasticsearch import Elasticsearch

from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader

from haystack.database.sql import SQLDocumentStore
from haystack.database.memory import InMemoryDocumentStore
from haystack.database.elasticsearch import ElasticsearchDocumentStore

@pytest.fixture(scope='session')
def elasticsearch_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('elasticsearch')


@pytest.fixture(scope="session")
def elasticsearch_fixture(elasticsearch_dir):
    # test if a ES cluster is already running. If not, download and start an ES instance locally.
    try:
        client = Elasticsearch(hosts=[{"host": "localhost"}])
        client.info()
    except:
        print("Downloading and starting an Elasticsearch instance for the tests ...")
        thetarfile = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.1-linux-x86_64.tar.gz"
        ftpstream = urllib.request.urlopen(thetarfile)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall(path=elasticsearch_dir)
        es_server = Popen([elasticsearch_dir / "elasticsearch-7.6.1/bin/elasticsearch"], stdout=PIPE, stderr=STDOUT)
        time.sleep(40)


@pytest.fixture(scope="session")
def xpdf_fixture():
    verify_installation = run(["pdftotext"], shell=True)
    if verify_installation.returncode == 127:
        commands = """ wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.02.tar.gz &&
                       tar -xvf xpdf-tools-linux-4.02.tar.gz && sudo cp xpdf-tools-linux-4.02/bin64/pdftotext /usr/local/bin"""
        run([commands], shell=True)

        verify_installation = run(["pdftotext -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """pdftotext is not installed. It is part of xpdf or poppler-utils software suite.
                 You can download for your OS from here: https://www.xpdfreader.com/download.html."""
            )


@pytest.fixture()
def test_docs_xs():
    return [
        {"text": "My name is Carla and I live in Berlin", "meta": {"meta_field": "test1", "name": "filename1"}},
        {"text": "My name is Paul and I live in New York", "meta": {"meta_field": "test2", "name": "filename2"}},
        {"text": "My name is Christelle and I live in Paris", "meta_field": "test3", "meta": {"name": "filename3"}}
        # last doc has meta_field at the top level for backward compatibility
    ]


@pytest.fixture(params=["farm", "transformers"])
def reader(request):
    if request.param == "farm":
        return FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad",
                          use_gpu=False, top_k_per_sample=5, num_processes=0)
    if request.param == "transformers":
        return TransformersReader(model="distilbert-base-uncased-distilled-squad",
                                  tokenizer="distilbert-base-uncased",
                                  use_gpu=-1)


@pytest.fixture(params=["sql", "memory", "elasticsearch"])
def document_store_with_docs(request, test_docs_xs, elasticsearch_fixture):
    if request.param == "sql":
        if os.path.exists("qa_test.db"):
            os.remove("qa_test.db")
        document_store = SQLDocumentStore(url="sqlite:///qa_test.db")
        document_store.write_documents(test_docs_xs)

    if request.param == "memory":
        document_store = InMemoryDocumentStore()
        document_store.write_documents(test_docs_xs)

    if request.param == "elasticsearch":
        # make sure we start from a fresh index
        client = Elasticsearch()
        client.indices.delete(index='haystack_test', ignore=[404])
        document_store = ElasticsearchDocumentStore(index="haystack_test")
        assert document_store.get_document_count() == 0
        document_store.write_documents(test_docs_xs)
        time.sleep(2)

    return document_store
