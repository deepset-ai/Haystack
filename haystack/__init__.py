# pylint: disable=wrong-import-position
# Logging is not configured here on purpose, see https://github.com/deepset-ai/haystack/issues/2485

import sys
from importlib import metadata

__version__: str = str(metadata.version("farm-haystack"))

from generalimport import generalimport, MissingOptionalDependency, FakeModule

generalimport(
    # "pydantic", # Required for all dataclasses
    # "tenacity",  # Probably needed because it's a decorator, to be evaluated
    # "pandas",
    "numpy",
    "requests",
    "transformers",
    "tokenizers",
    "PIL",
    "yaml",
    "torch",
    "protobuf",
    "nltk",
    "rank_bm25",
    "sklearn",
    "dill",
    "tqdm",
    "networkx",
    "mmh3",
    "quantulum3",
    "posthog",
    "azure",
    "huggingface_hub",
    "tenacity",
    "sseclient",
    "boilerpy3",
    "more_itertools",
    "docx",
    "langdetect",
    "tika",
    "sentence_transformers",
    "elasticsearch",
    "tiktoken",
    "jsonschema",
    "canals",
    "events",
    "sqlalchemy",
    "psycopg2",
    "faiss",
    "pymilvus",
    "weaviate",
    "pinecone",
    "SPARQLWrapper",
    "rdflib",
    "opensearchpy",
    "whisper",
    "beir",
    "selenium",
    "webdriver_manager",
    "beautifulsoup4",
    "markdown",
    "frontmatter",
    "magic",
    "fitz",
    "pytesseract",
    "pdf2image",
    "onnxruntime",
    "onnxruntime_tools",
    "scipy",
    "rapidfuzz",
    "seqeval",
    "mlflow",
    "ray",
    "aiorwlock",
)

from haystack.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult, TableCell
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline
from haystack.environment import set_pytorch_secure_model_loading


# Enables torch's secure model loading through setting an env var.
# Does not use torch.
set_pytorch_secure_model_loading()
