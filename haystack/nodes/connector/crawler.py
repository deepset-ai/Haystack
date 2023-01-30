from typing import Callable, List, Optional, Dict, Tuple, Union, Any

import os
import re
import sys
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
import hashlib

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
    from selenium import webdriver
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "crawler", ie)

from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.errors import NodeError


logger = logging.getLogger(__name__)


class Crawler(BaseComponent):
    """
    Crawl texts from a website so that we can use them later in Haystack as a corpus for search / question answering etc.

    **Example:**
    ```python
    from haystack.nodes.connector import Crawler

    crawler = Crawler(output_dir="crawled_files")
    # crawl Haystack docs, i.e. all pages that include haystack.deepset.ai/overview/
    docs = crawler.crawl(urls=["https://haystack.deepset.ai/overview/get-started"],
                         filter_urls= ["haystack.deepset.ai/overview/"])
    ```
    """

    outgoing_edges = 1

    def __init__(
        self,
        output_dir: str,
        urls: Optional[List[str]] = None,
        crawler_depth: int = 1,
        filter_urls: Optional[List] = None,
        overwrite_existing_files=True,
        id_hash_keys: Optional[List[str]] = None,
        extract_hidden_text=True,
        loading_wait_time: Optional[int] = None,
        crawler_naming_function: Optional[Callable[[str, str], str]] = None,
        webdriver_options: Optional[List[str]] = None,
    ):
        """
        Init object with basic params for crawling (can be overwritten later).

        :param output_dir: Path for the directory to store files
        :param urls: List of http(s) address(es) (can also be supplied later when calling crawl())
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
            0: Only initial list of urls
            1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
            All URLs not matching at least one of the regular expressions will be dropped.
        :param overwrite_existing_files: Whether to overwrite existing files in output_dir with new content
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param extract_hidden_text: Whether to extract the hidden text contained in page.
            E.g. the text can be inside a span with style="display: none"
        :param loading_wait_time: Seconds to wait for page loading before scraping. Recommended when page relies on
            dynamic DOM manipulations. Use carefully and only when needed. Crawler will have scraping speed impacted.
            E.g. 2: Crawler will wait 2 seconds before scraping page
        :param crawler_naming_function: A function mapping the crawled page to a file name.
            By default, the file name is generated from the processed page url (string compatible with Mac, Unix and Windows paths) and the last 6 digits of the MD5 sum of this unprocessed page url.
            E.g. 1) crawler_naming_function=lambda url, page_content: re.sub("[<>:'/\\|?*\0 ]", "_", link)
                    This example will generate a file name from the url by replacing all characters that are not allowed in file names with underscores.
                 2) crawler_naming_function=lambda url, page_content: hashlib.md5(f"{url}{page_content}".encode("utf-8")).hexdigest()
                    This example will generate a file name from the url and the page content by using the MD5 hash of the concatenation of the url and the page content.
        :param webdriver_options: A list of options to send to Selenium webdriver. If none is provided,
            Crawler uses, as a default option, a reasonable selection for operating locally, on restricted docker containers,
            and avoids using GPU.
            Crawler always appends the following option: "--headless"
            For example: 1) ["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage", "--single-process"]
                    These are the default options which disable GPU, disable shared memory usage
                    and spawn a single process.
                 2) ["--no-sandbox"]
                    This option disables the sandbox, which is required for running Chrome as root.
            See [Chrome Web Driver Options](https://selenium-python.readthedocs.io/api.html#module-selenium.webdriver.chrome.options) for more details.
        """
        super().__init__()

        IN_COLAB = "google.colab" in sys.modules
        IN_AZUREML = os.environ.get("AZUREML_ENVIRONMENT_IMAGE", None) == "True"
        IS_ROOT = sys.platform not in ["win32", "cygwin"] and os.geteuid() == 0

        if webdriver_options is None:
            webdriver_options = ["--headless", "--disable-gpu", "--disable-dev-shm-usage", "--single-process"]
        elif "--headless" not in webdriver_options:
            webdriver_options.append("--headless")

        if IS_ROOT and "--no-sandbox" not in webdriver_options:
            webdriver_options.append("--no-sandbox")

        if (IN_COLAB or IN_AZUREML) and "--disable-dev-shm-usage" not in webdriver_options:
            webdriver_options.append("--disable-dev-shm-usage")

        options = Options()
        for option in webdriver_options:
            options.add_argument(option)

        if IN_COLAB:
            try:
                self.driver = webdriver.Chrome(service=Service("chromedriver"), options=options)
            except WebDriverException as exc:
                raise NodeError(
                    """
        \'chromium-driver\' needs to be installed manually when running colab. Follow the below given commands:
                        %%shell
                        cat > /etc/apt/sources.list.d/debian.list <<'EOF'
                        deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster.gpg] http://deb.debian.org/debian buster main
                        deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster-updates.gpg] http://deb.debian.org/debian buster-updates main
                        deb [arch=amd64 signed-by=/usr/share/keyrings/debian-security-buster.gpg] http://deb.debian.org/debian-security buster/updates main
                        EOF

                        apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517
                        apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138
                        apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A
                        apt-key export 77E11517 | gpg --dearmour -o /usr/share/keyrings/debian-buster.gpg
                        apt-key export 22F3D138 | gpg --dearmour -o /usr/share/keyrings/debian-buster-updates.gpg
                        apt-key export E562B32A | gpg --dearmour -o /usr/share/keyrings/debian-security-buster.gpg

                        cat > /etc/apt/preferences.d/chromium.pref << 'EOF'
                        Package: *
                        Pin: release a=eoan
                        Pin-Priority: 500


                        Package: *
                        Pin: origin "deb.debian.org"
                        Pin-Priority: 300


                        Package: chromium*
                        Pin: origin "deb.debian.org"
                        Pin-Priority: 700
                        EOF

                        apt-get update
                        apt-get install chromium chromium-driver
        If it has already been installed, please check if it has been copied to the right directory i.e. to \'/usr/bin\'"""
                ) from exc
        else:
            logger.info("'chrome-driver' will be automatically installed.")
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.urls = urls
        self.output_dir = output_dir
        self.crawler_depth = crawler_depth
        self.filter_urls = filter_urls
        self.overwrite_existing_files = overwrite_existing_files
        self.id_hash_keys = id_hash_keys
        self.extract_hidden_text = extract_hidden_text
        self.loading_wait_time = loading_wait_time
        self.crawler_naming_function = crawler_naming_function

    def __del__(self):
        self.driver.quit()

    def crawl(
        self,
        output_dir: Union[str, Path, None] = None,
        urls: Optional[List[str]] = None,
        crawler_depth: Optional[int] = None,
        filter_urls: Optional[List] = None,
        overwrite_existing_files: Optional[bool] = None,
        id_hash_keys: Optional[List[str]] = None,
        extract_hidden_text: Optional[bool] = None,
        loading_wait_time: Optional[int] = None,
        crawler_naming_function: Optional[Callable[[str, str], str]] = None,
    ) -> List[Path]:
        """
        Craw URL(s), extract the text from the HTML, create a Haystack Document object out of it and save it (one JSON
        file per URL, including text and basic meta data).
        You can optionally specify via `filter_urls` to only crawl URLs that match a certain pattern.
        All parameters are optional here and only meant to overwrite instance attributes at runtime.
        If no parameters are provided to this method, the instance attributes that were passed during __init__ will be used.

        :param output_dir: Path for the directory to store files
        :param urls: List of http addresses or single http address
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :param overwrite_existing_files: Whether to overwrite existing files in output_dir with new content
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param loading_wait_time: Seconds to wait for page loading before scraping. Recommended when page relies on
            dynamic DOM manipulations. Use carefully and only when needed. Crawler will have scraping speed impacted.
            E.g. 2: Crawler will wait 2 seconds before scraping page
        :param crawler_naming_function: A function mapping the crawled page to a file name.
            By default, the file name is generated from the processed page url (string compatible with Mac, Unix and Windows paths) and the last 6 digits of the MD5 sum of this unprocessed page url.
            E.g. 1) crawler_naming_function=lambda url, page_content: re.sub("[<>:'/\\|?*\0 ]", "_", link)
                    This example will generate a file name from the url by replacing all characters that are not allowed in file names with underscores.
                 2) crawler_naming_function=lambda url, page_content: hashlib.md5(f"{url}{page_content}".encode("utf-8")).hexdigest()
                    This example will generate a file name from the url and the page content by using the MD5 hash of the concatenation of the url and the page content.

        :return: List of paths where the crawled webpages got stored
        """
        # use passed params or fallback to instance attributes
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        urls = urls or self.urls
        if urls is None:
            raise ValueError("Got no urls to crawl. Set `urls` to a list of URLs in __init__(), crawl() or run(). `")
        output_dir = output_dir or self.output_dir
        filter_urls = filter_urls or self.filter_urls
        if overwrite_existing_files is None:
            overwrite_existing_files = self.overwrite_existing_files
        if crawler_depth is None:
            crawler_depth = self.crawler_depth
        if extract_hidden_text is None:
            extract_hidden_text = self.extract_hidden_text
        if loading_wait_time is None:
            loading_wait_time = self.loading_wait_time
        if crawler_naming_function is None:
            crawler_naming_function = self.crawler_naming_function

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        file_paths: list = []
        is_not_empty = len(list(output_dir.rglob("*"))) > 0
        if is_not_empty and not overwrite_existing_files:
            logger.info(
                "Found data stored in `%s`. Delete this first if you really want to fetch new data.", output_dir
            )
        else:
            logger.info("Fetching from %s to `%s`", urls, output_dir)

            # Start by writing out the initial list of urls
            if filter_urls:
                pattern = re.compile("|".join(filter_urls))
                for url in urls:
                    if pattern.search(url):
                        file_paths += self._write_to_files(
                            [url],
                            output_dir=output_dir,
                            extract_hidden_text=extract_hidden_text,
                            loading_wait_time=loading_wait_time,
                            crawler_naming_function=crawler_naming_function,
                        )
            else:
                file_paths += self._write_to_files(
                    urls,
                    output_dir=output_dir,
                    extract_hidden_text=extract_hidden_text,
                    loading_wait_time=loading_wait_time,
                    crawler_naming_function=crawler_naming_function,
                )
            # follow one level of sublinks if requested
            if crawler_depth == 1:
                sub_links: Dict[str, List] = {}
                for url_ in urls:
                    already_found_links: List = list(sum(list(sub_links.values()), []))
                    sub_links[url_] = list(
                        self._extract_sublinks_from_url(
                            base_url=url_,
                            filter_urls=filter_urls,
                            already_found_links=already_found_links,
                            loading_wait_time=loading_wait_time,
                        )
                    )
                for url, extracted_sublink in sub_links.items():
                    file_paths += self._write_to_files(
                        extracted_sublink,
                        output_dir=output_dir,
                        base_url=url,
                        id_hash_keys=id_hash_keys,
                        extract_hidden_text=extract_hidden_text,
                        loading_wait_time=loading_wait_time,
                        crawler_naming_function=crawler_naming_function,
                    )

        return file_paths

    def _write_to_files(
        self,
        urls: List[str],
        output_dir: Path,
        extract_hidden_text: bool,
        base_url: Optional[str] = None,
        id_hash_keys: Optional[List[str]] = None,
        loading_wait_time: Optional[int] = None,
        crawler_naming_function: Optional[Callable[[str, str], str]] = None,
    ) -> List[Path]:
        paths = []
        for link in urls:
            logger.info("writing contents from '%s'", link)
            self.driver.get(link)
            if loading_wait_time is not None:
                time.sleep(loading_wait_time)
            el = self.driver.find_element(by=By.TAG_NAME, value="body")
            if extract_hidden_text:
                text = el.get_attribute("textContent")
            else:
                text = el.text

            data: Dict[str, Any] = {}
            data["meta"] = {"url": link}
            if base_url:
                data["meta"]["base_url"] = base_url
            data["content"] = text
            if id_hash_keys:
                data["id_hash_keys"] = id_hash_keys
            document = Document.from_dict(data)

            if crawler_naming_function is not None:
                file_name_prefix = crawler_naming_function(link, text)
            else:
                file_name_link = re.sub("[<>:'/\\|?*\0 ]", "_", link[:129])
                file_name_hash = hashlib.md5(f"{link}".encode("utf-8")).hexdigest()
                file_name_prefix = f"{file_name_link}_{file_name_hash[-6:]}"

            file_path = output_dir / f"{file_name_prefix}.json"

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(document.to_dict(), f)
            except Exception:
                logging.exception(
                    "Crawler can't save the content of '%s' under '%s'. "
                    "This webpage will be skipped, but links from this page will still be crawled. "
                    "Make sure the path above is accessible and the file name is valid. "
                    "If the file name is invalid, consider setting 'crawler_naming_function' to another function.",
                    link,
                    file_path,
                )

            paths.append(file_path)

        return paths

    def run(  # type: ignore
        self,
        output_dir: Union[str, Path, None] = None,
        urls: Optional[List[str]] = None,
        crawler_depth: Optional[int] = None,
        filter_urls: Optional[List] = None,
        overwrite_existing_files: Optional[bool] = None,
        return_documents: Optional[bool] = False,
        id_hash_keys: Optional[List[str]] = None,
        extract_hidden_text: Optional[bool] = True,
        loading_wait_time: Optional[int] = None,
        crawler_naming_function: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[Dict[str, Union[List[Document], List[Path]]], str]:
        """
        Method to be executed when the Crawler is used as a Node within a Haystack pipeline.

        :param output_dir: Path for the directory to store files
        :param urls: List of http addresses or single http address
        :param crawler_depth: How many sublinks to follow from the initial list of URLs. Current options:
                              0: Only initial list of urls
                              1: Follow links found on the initial URLs (but no further)
        :param filter_urls: Optional list of regular expressions that the crawled URLs must comply with.
                           All URLs not matching at least one of the regular expressions will be dropped.
        :param overwrite_existing_files: Whether to overwrite existing files in output_dir with new content
        :param return_documents:  Return json files content
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are
            not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]).
            In this case the id will be generated by using the content and the defined metadata.
        :param extract_hidden_text: Whether to extract the hidden text contained in page.
            E.g. the text can be inside a span with style="display: none"
        :param loading_wait_time: Seconds to wait for page loading before scraping. Recommended when page relies on
            dynamic DOM manipulations. Use carefully and only when needed. Crawler will have scraping speed impacted.
            E.g. 2: Crawler will wait 2 seconds before scraping page
        :param crawler_naming_function: A function mapping the crawled page to a file name.
            By default, the file name is generated from the processed page url (string compatible with Mac, Unix and Windows paths) and the last 6 digits of the MD5 sum of this unprocessed page url.
            E.g. 1) crawler_naming_function=lambda url, page_content: re.sub("[<>:'/\\|?*\0 ]", "_", link)
                    This example will generate a file name from the url by replacing all characters that are not allowed in file names with underscores.
                 2) crawler_naming_function=lambda url, page_content: hashlib.md5(f"{url}{page_content}".encode("utf-8")).hexdigest()
                    This example will generate a file name from the url and the page content by using the MD5 hash of the concatenation of the url and the page content.

        :return: Tuple({"paths": List of filepaths, ...}, Name of output edge)
        """

        file_paths = self.crawl(
            urls=urls,
            output_dir=output_dir,
            crawler_depth=crawler_depth,
            filter_urls=filter_urls,
            overwrite_existing_files=overwrite_existing_files,
            extract_hidden_text=extract_hidden_text,
            loading_wait_time=loading_wait_time,
            crawler_naming_function=crawler_naming_function,
        )
        results: Dict[str, Union[List[Document], List[Path]]] = {}
        if return_documents:
            crawled_data = []
            for _file in file_paths:
                with open(_file.absolute(), "r") as read_file:
                    document = json.load(read_file)
                    document["id_hash_keys"] = id_hash_keys
                    crawled_data.append(Document.from_dict(document))
            results = {"documents": crawled_data}
        else:
            results = {"paths": file_paths}

        return results, "output_1"

    def run_batch(  # type: ignore
        self,
        output_dir: Union[str, Path, None] = None,
        urls: Optional[List[str]] = None,
        crawler_depth: Optional[int] = None,
        filter_urls: Optional[List] = None,
        overwrite_existing_files: Optional[bool] = None,
        return_documents: Optional[bool] = False,
        id_hash_keys: Optional[List[str]] = None,
        extract_hidden_text: Optional[bool] = True,
        loading_wait_time: Optional[int] = None,
        crawler_naming_function: Optional[Callable[[str, str], str]] = None,
    ):
        return self.run(
            output_dir=output_dir,
            urls=urls,
            crawler_depth=crawler_depth,
            filter_urls=filter_urls,
            overwrite_existing_files=overwrite_existing_files,
            return_documents=return_documents,
            id_hash_keys=id_hash_keys,
            extract_hidden_text=extract_hidden_text,
            loading_wait_time=loading_wait_time,
            crawler_naming_function=crawler_naming_function,
        )

    @staticmethod
    def _is_internal_url(base_url: str, sub_link: str) -> bool:
        base_url_ = urlparse(base_url)
        sub_link_ = urlparse(sub_link)
        return base_url_.scheme == sub_link_.scheme and base_url_.netloc == sub_link_.netloc

    @staticmethod
    def _is_inpage_navigation(base_url: str, sub_link: str) -> bool:
        base_url_ = urlparse(base_url)
        sub_link_ = urlparse(sub_link)
        return base_url_.path == sub_link_.path and base_url_.netloc == sub_link_.netloc

    def _extract_sublinks_from_url(
        self,
        base_url: str,
        filter_urls: Optional[List] = None,
        already_found_links: Optional[List] = None,
        loading_wait_time: Optional[int] = None,
    ) -> set:

        self.driver.get(base_url)
        if loading_wait_time is not None:
            time.sleep(loading_wait_time)
        a_elements = self.driver.find_elements(by=By.XPATH, value="//a[@href]")
        sub_links = set()

        filter_pattern = re.compile("|".join(filter_urls)) if filter_urls is not None else None

        for i in a_elements:
            try:
                sub_link = i.get_attribute("href")
            except StaleElementReferenceException:
                logger.error(
                    "The crawler couldn't find the link anymore. It has probably been removed from DOM by JavaScript."
                )
                continue

            if not (already_found_links and sub_link in already_found_links):
                if self._is_internal_url(base_url=base_url, sub_link=sub_link) and (
                    not self._is_inpage_navigation(base_url=base_url, sub_link=sub_link)
                ):
                    if filter_pattern is not None:

                        if filter_pattern.search(sub_link):
                            sub_links.add(sub_link)
                    else:
                        sub_links.add(sub_link)

        return sub_links
