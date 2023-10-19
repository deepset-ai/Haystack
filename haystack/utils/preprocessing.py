from typing import Callable, Dict, List, Optional

import re
import logging
from pathlib import Path

from haystack.schema import Document


logger = logging.getLogger(__name__)


def convert_files_to_docs(
    dir_path: Optional[str] = None,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    encoding: Optional[str] = None,
    id_hash_keys: Optional[List[str]] = None,
    file_paths: Optional[List[Path]] = None,
) -> List[Document]:
    """
    Convert files (.txt, .pdf, .docx) to Documents that can be written to a Document Store.

    Files can be specified by giving a directory path, a list of file paths, or both. If a directory path is given then
    all files with the allowed suffixes in the directory's subdirectories will be converted.

    :param dir_path: The path of a directory that contains Files to be converted, including in its subdirectories.
    :param clean_func: A custom cleaning function that gets applied to each Document (input: str, output: str).
    :param split_paragraphs: Whether to split text by paragraph.
    :param encoding: Character encoding to use when converting pdf documents.
    :param id_hash_keys: A list of Document attribute names from which the Document ID should be hashed from.
            Useful for generating unique IDs even if the Document contents are identical.
            To ensure you don't have duplicate Documents in your Document Store if texts are
            not unique, you can modify the metadata and pass [`"content"`, `"meta"`] to this field.
            If you do this, the Document ID will be generated by using the content and the defined metadata.
    :param file_paths: A list of paths of Files to be converted.
    """
    # Importing top-level causes a circular import
    from haystack.nodes.file_converter import BaseConverter, DocxToTextConverter, PDFToTextConverter, TextConverter

    if dir_path is None and file_paths is None:
        raise ValueError("At least one of dir_path or file_paths must be set.")
    if file_paths is None:
        file_paths = []
    if dir_path is not None:
        file_paths = file_paths + list(Path(dir_path).glob("**/*"))

    allowed_suffixes = [".pdf", ".txt", ".docx"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file %s as type %s is not supported here. "
                "See haystack.file_converter for support of more file types",
                path,
                file_suffix,
            )

    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxToTextConverter()

    documents = []
    for suffix, paths in suffix2paths.items():
        for path in paths:
            logger.info("Converting %s", path)
            # PDFToTextConverter, TextConverter, and DocxToTextConverter return a list containing a single Document
            document = suffix2converter[suffix].convert(
                file_path=path, meta=None, encoding=encoding, id_hash_keys=id_hash_keys
            )[0]
            text = document.content

            if clean_func:
                text = clean_func(text)

            if split_paragraphs:
                for para in text.split("\n\n"):
                    if not para.strip():  # skip empty paragraphs
                        continue
                    documents.append(Document(content=para, meta={"name": path.name}, id_hash_keys=id_hash_keys))
            else:
                documents.append(Document(content=text, meta={"name": path.name}, id_hash_keys=id_hash_keys))

    return documents


def tika_convert_files_to_docs(
    dir_path: Optional[str] = None,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    merge_short: bool = True,
    merge_lowercase: bool = True,
    id_hash_keys: Optional[List[str]] = None,
    file_paths: Optional[List[Path]] = None,
) -> List[Document]:
    """
    Convert files (.txt, .pdf) to Documents that can be written to a Document Store.

    Files can be specified by giving a directory path, a list of file paths, or both. If a directory path is given then
    all files with the allowed suffixes in the directory's subdirectories will be converted.

    :param merge_lowercase: Whether to convert merged paragraphs to lowercase.
    :param merge_short: Whether to allow merging of short paragraphs
    :param dir_path: The path of a directory that contains Files to be converted, including in its subdirectories.
    :param clean_func: A custom cleaning function that gets applied to each doc (input: str, output:str).
    :param split_paragraphs: Whether to split text by paragraphs.
    :param id_hash_keys: A list of Document attribute names from which the Document ID should be hashed from.
            Useful for generating unique IDs even if the Document contents are identical.
            To ensure you don't have duplicate Documents in your Document Store if texts are
            not unique, you can modify the metadata and pass [`"content"`, `"meta"`] to this field.
            If you do this, the Document ID will be generated by using the content and the defined metadata.
    :param file_paths: A list of paths of Files to be converted.
    """
    try:
        from haystack.nodes.file_converter import TikaConverter
    except Exception as ex:
        logger.error("Tika not installed. Please install tika and try again. Error: %s", ex)
        raise ex
    converter = TikaConverter()

    if dir_path is None and file_paths is None:
        raise ValueError("At least one of dir_path or file_paths must be set.")
    if file_paths is None:
        file_paths = []
    if dir_path is not None:
        file_paths = file_paths + list(Path(dir_path).glob("**/*"))

    allowed_suffixes = [".pdf", ".txt"]
    file_paths_to_convert: List[Path] = []

    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            file_paths_to_convert.append(path)
        elif not path.is_dir():
            logger.warning(
                "Skipped file %s as type %s is not supported here. "
                "See haystack.file_converter for support of more file types",
                path,
                file_suffix,
            )

    documents = []
    for path in file_paths_to_convert:
        logger.info("Converting %s", path)
        # TikaConverter returns a list containing a single Document
        document = converter.convert(path)[0]
        meta = document.meta or {}
        meta["name"] = path.name
        text = document.content
        pages = text.split("\f")

        if split_paragraphs:
            if pages:
                paras = pages[0].split("\n\n")
                # pop the last paragraph from the first page
                last_para = paras.pop(-1) if paras else ""
                for page in pages[1:]:
                    page_paras = page.split("\n\n")
                    # merge the last paragraph in previous page to the first paragraph in this page
                    if page_paras:
                        page_paras[0] = last_para + " " + page_paras[0]
                        last_para = page_paras.pop(-1)
                        paras += page_paras
                if last_para:
                    paras.append(last_para)
                if paras:
                    last_para = ""
                    for para in paras:
                        para = para.strip()
                        if not para:
                            continue

                        # this paragraph is less than 10 characters or 2 words
                        para_is_short = len(para) < 10 or len(re.findall(r"\s+", para)) < 2
                        # this paragraph starts with a lower case and last paragraph does not end with a punctuation
                        para_is_lowercase = (
                            para and para[0].islower() and last_para and last_para[-1] not in r'.?!"\'\]\)'
                        )

                        # merge paragraphs to improve qa
                        if (merge_short and para_is_short) or (merge_lowercase and para_is_lowercase):
                            last_para += " " + para
                        else:
                            if last_para:
                                documents.append(Document(content=last_para, meta=meta, id_hash_keys=id_hash_keys))
                            last_para = para
                    # don't forget the last one
                    if last_para:
                        documents.append(Document(content=last_para, meta=meta, id_hash_keys=id_hash_keys))

        else:
            if clean_func:
                text = clean_func(text)
            documents.append(Document(content=text, meta=meta, id_hash_keys=id_hash_keys))

    return documents
