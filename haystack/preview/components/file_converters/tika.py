import logging
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from haystack.preview.lazy_imports import LazyImport
from haystack.preview import component, Document, default_to_dict, default_from_dict


with LazyImport("Run 'pip install tika'") as tika_import:
    from tika import parser as tika_parser

logger = logging.getLogger(__name__)


@component
class TikaDocumentConverter:
    """
    A component for converting files of different types (pdf, docx, html, etc.) to Documents.
    This component uses [Apache Tika](https://tika.apache.org/) for parsing the files and, therefore,
    requires a running Tika server.
    """

    def __init__(self, tika_url: str = "http://localhost:9998/tika", id_hash_keys: Optional[List[str]] = None):
        """
        Create a TikaDocumentConverter component.

        :param tika_url: URL of the Tika server. Default: `"http://localhost:9998/tika"`
        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. If you want to ensure you don't have duplicate Documents in your DocumentStore but texts are not
            unique, you can pass the name of the metadata to use when building the document ID (like
            `["text", "category"]`) to this field. In this case, the ID will be generated by using the text and the content of the
            `category` field. Default: `None`
        """
        tika_import.check()
        self.tika_url = tika_url
        self.id_hash_keys = id_hash_keys or []

    @component.output_types(documents=List[Document])
    def run(self, paths: List[Union[str, Path]], id_hash_keys: Optional[List[str]] = None):
        """
        Convert files to Documents.

        :param paths: A list of paths to the files to convert.
        :param id_hash_keys: Generate the Document ID from a custom list of strings that refer to the Document's
            attributes. If you want to ensure you don't have duplicate Documents in your DocumentStore but texts are not
            unique, you can pass the name of the metadata to use when building the document ID (like
            `["text", "category"]`) to this field. In this case, the ID will be generated by using the text and the
            content of the `category` field.
            If not set, the id_hash_keys passed to the constructor will be used.
            Default: `None`

        """
        id_hash_keys = id_hash_keys or self.id_hash_keys

        documents = []
        for path in paths:
            path = Path(path)
            try:
                parsed_file = tika_parser.from_file(path.as_posix(), self.tika_url)
                extracted_text = parsed_file["content"]
                if not extracted_text:
                    logger.warning("Skipping file at '%s' as Tika was not able to extract any content.", str(path))
                    continue
                document = Document(text=extracted_text, id_hash_keys=id_hash_keys)
                documents.append(document)
            except Exception as e:
                logger.error("Could not convert file at '%s' to Document. Error: %s", str(path), e)

        return {"documents": documents}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, tika_url=self.tika_url, id_hash_keys=self.id_hash_keys)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TikaDocumentConverter":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)
