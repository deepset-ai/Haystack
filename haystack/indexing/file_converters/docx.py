from haystack.indexing.file_converters.base import BaseConverter
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocxToTextConverter(BaseConverter):
    def __init__(self):
        try:
            import docx
        except ImportError as e:
            raise Exception(
                """python-docx is not installed.
                
                Please run !pip install python-docx
                
                   You can find more details here: https://python-docx.readthedocs.io/en/latest/
                """
            )
    def extract_pages(self, file_path: Path) -> list[str]:
        import docx
        doc = docx.Document(file_path)  # Creating word reader object.
        fullText = []
        for para in doc.paragraphs:
          fullText.append(para.text)
        return fullText
