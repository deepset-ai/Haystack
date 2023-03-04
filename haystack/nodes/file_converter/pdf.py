import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from more_itertools import divide

try:
    import fitz
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "pdf", ie)

from haystack.nodes.file_converter.base import BaseConverter
from haystack.schema import Document

logger = logging.getLogger(__name__)


class PDFToTextConverter(BaseConverter):
    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = None,
        id_hash_keys: Optional[List[str]] = None,
        encoding: Optional[str] = None,
        keep_physical_layout: Optional[bool] = None,
        sort_by_position: bool = False,
        ocr: Optional[Literal["auto", "full"]] = None,
        ocr_language: str = "eng",
        multiprocessing: Union[bool, int] = True,
    ) -> None:
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param encoding: This parameter is being deprecated.
                         It will be automatically detected by PyMuPDF.
        :param keep_physical_layout: This parameter is being deprecated.
        :param sort_by_position: Specifies whether to sort the extracted text by positional coordinates or logical reading order.
                        If set to True, the text is sorted first by vertical position, and then by horizontal position.
                        If set to False (default), the logical reading order in the PDF is used.
        :param ocr: Specifies whether to use OCR to extract text from images in the PDF. If set to "auto", OCR is used only to extract text
                    from images and integrate into the existing text. If set to "full", OCR is used to extract text from the entire PDF.
        :param ocr_language: Specifies the language to use for OCR. The default language is English, which language code is `eng`.
                For a list of supported languages and the respective codes access https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.
                You can combine multiple languages by passing a string with the language codes separated by `+`. For example, to use English and German, pass `eng+deu`.
        :param multiprocessing: We use multiprocessing to speed up PyMuPDF conversion, you can disable it by setting it to False.
                                If set to True (the default value), the total number of cores is used. To specify the number of cores to use, set it to an integer.
        """
        super().__init__(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages, id_hash_keys=id_hash_keys
        )

        self.sort_by_position = sort_by_position
        self.multiprocessing = multiprocessing
        self.ocr = ocr
        self.ocr_language = ocr_language

        if ocr is not None:
            self._check_tessdata()

        if encoding:
            warnings.warn(
                "The encoding parameter is being deprecated. It will be automatically detected by PyMuPDF.",
                DeprecationWarning,
            )

        if keep_physical_layout:
            warnings.warn("The keep_physical_layout parameter is being deprecated.", DeprecationWarning)

    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, Any]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = None,
        id_hash_keys: Optional[List[str]] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        keep_physical_layout: Optional[bool] = None,
        sort_by_position: Optional[bool] = None,
        ocr: Optional[Literal["auto", "full"]] = None,
        ocr_language: Optional[str] = None,
        multiprocessing: Optional[Union[bool, int]] = None,
    ) -> List[Document]:
        """
        Extract text from a PDF file and convert it to a Document.
        :param file_path: Path to the .pdf file you want to convert
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: This parameter is being deprecated.
                         It will be automatically detected by PyMuPDF.
        :param keep_physical_layout: This parameter is being deprecated.
        :param sort_by_position: Specifies whether to sort the extracted text by positional coordinates or logical reading order.
                        If set to True, the text is sorted first by vertical position, and then by horizontal position.
                        If set to False (default), the logical reading order in the PDF is used.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param start_page: The page number where to start the conversion
        :param end_page: The page number where to end the conversion.
        :param ocr: Specifies whether to use OCR to extract text from images in the PDF. If set to "auto", OCR is used only to extract text
                    from images and integrate into the existing text. If set to "full", OCR is used to extract text from the entire PDF.
                    To use this feature you must install Tesseract-OCR. For more information, see https://github.com/tesseract-ocr/tesseract#installing-tesseract.
        :param ocr_language: Specifies the language to use for OCR. The default language is English, which language code is `eng`.
                For a list of supported languages and the respective codes access https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.
                You can combine multiple languages by passing a string with the language codes separated by `+`. For example, to use English and German, pass `eng+deu`.
        :param multiprocessing: We use multiprocessing to speed up PyMuPDF conversion, you can disable it by setting it to False.
                                If set to None (the default value), the value defined in the class initialization is used.
                                If set to True, the total number of cores is used. To specify the number of cores to use, set it to an integer.
        """
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys
        if multiprocessing is None:
            multiprocessing = self.multiprocessing
        if sort_by_position is None:
            sort_by_position = self.sort_by_position
        if ocr is None:
            ocr = self.ocr
        if ocr_language is None:
            ocr_language = self.ocr_language

        if encoding:
            warnings.warn(
                "The encoding parameter is being deprecated. It will be automatically detected by PyMuPDF.",
                DeprecationWarning,
            )

        if keep_physical_layout:
            warnings.warn("The keep_physical_layout parameter is being deprecated.", DeprecationWarning)

        if ocr is not None:
            self._check_tessdata()

        pages = self._read_pdf(
            file_path,
            sort_by_position=sort_by_position,
            start_page=start_page,
            end_page=end_page,
            ocr=ocr,
            ocr_language=ocr_language,
            multiprocessing=multiprocessing,
        )

        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                        logger.debug("Removing line '%s' from %s", line, file_path)
                        continue
                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text, valid_languages):
                logger.warning(
                    "The language for %s is not one of %s. The file may not have "
                    "been decoded in the correct text format.",
                    file_path,
                    valid_languages,
                )

        text = "\f".join(cleaned_pages)
        document = Document(content=text, meta=meta, id_hash_keys=id_hash_keys)
        return [document]

    def _check_tessdata(self):
        if os.getenv("TESSDATA_PREFIX") is None:
            logger.error(
                """
                To enable OCR support via PDFToTextConverter, you need to install Tesseract:
                    - Windows: choco install tesseract-ocr
                    - Linux (Ubuntu): sudo apt-get install tesseract-ocr
                    - Mac: brew install tesseract
                After that, you need to set the environment variable TESSDATA_PREFIX to the path
                of your Tesseract data directory. Typically this is:
                    - Windows: C:\\Program Files\\Tesseract-OCR\\tessdata
                    - Linux (Ubuntu): /usr/share/tesseract-ocr/4.00/tessdata
                    - Mac:  /usr/local/Cellar/tesseract/5.3.0_1/share/tessdata
                """
            )

    def _get_text_parallel(self, page_mp):
        idx, filename, parts, sort_by_position, ocr, ocr_language = page_mp

        doc = fitz.open(filename)

        text = ""
        for i in parts[idx]:
            page = doc[i]
            partial_tp = None
            if ocr is not None:
                partial_tp = page.get_textpage_ocr(
                    flags=0, full=True if ocr == "full" else False, dpi=300, language=ocr_language
                )
            text += page.get_text("text", textpage=partial_tp, sort=sort_by_position) + "\f"

        return text

    def _read_pdf(
        self,
        file_path: Path,
        ocr_language: str,
        sort_by_position: bool = False,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        ocr: Optional[Literal["auto", "full"]] = None,
        multiprocessing: Optional[Union[bool, int]] = None,
    ) -> List[str]:
        """
        Extract pages from the pdf file at file_path.

        :param file_path: path of the pdf file
        :param sort_by_position: Specifies whether to sort the extracted text by positional coordinates or logical reading order.
                        If set to True, the text is sorted first by vertical position, and then by horizontal position.
                        If set to False (default), the logical reading order in the PDF is used.
        :param start_page: The page number where to start the conversion, starting from 1.
        :param end_page: The page number where to end the conversion.
        :param encoding: This parameter is being deprecated.
                         It will be automatically detected by PyMuPDF.
        :param multiprocessing: We use multiprocessing to speed up PyMuPDF conversion, you can disable it by setting it to False.
                                If set to None (the default value), the value defined in the class initialization is used.
                                If set to True, the total number of cores is used. To specify the number of cores to use, set it to an integer.
        """
        if start_page is None:
            start_page = 0
        else:
            start_page = start_page - 1

        doc = fitz.open(file_path)
        page_count = int(doc.page_count)

        if end_page is None or (end_page is not None and end_page > page_count):
            end_page = page_count

        document = ""

        if not multiprocessing:
            for i in range(start_page, end_page):
                page = doc[i]
                partial_tp = None
                if ocr is not None:
                    partial_tp = page.get_textpage_ocr(
                        flags=0, full=True if ocr == "full" else False, dpi=300, language=ocr_language
                    )
                document += page.get_text("text", textpage=partial_tp, sort=sort_by_position) + "\f"
        else:
            cpu = cpu_count() if isinstance(multiprocessing, bool) else multiprocessing
            page_list = [i for i in range(start_page, end_page)]
            cpu = cpu if len(page_list) > cpu else len(page_list)
            parts = divide(cpu, page_list)
            pages_mp = [(i, file_path, parts, sort_by_position, ocr, ocr_language) for i in range(cpu)]

            with ProcessPoolExecutor(max_workers=cpu) as pool:
                results = pool.map(self._get_text_parallel, pages_mp)
                for page in results:
                    document += page

        document = "\f" * start_page + document  # tracking skipped pages for correct page numbering
        pages = document.split("\f")
        pages = pages[:-1]  # the last page in the split is always empty.

        return pages
