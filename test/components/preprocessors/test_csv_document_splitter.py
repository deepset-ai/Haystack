# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd
from io import StringIO
from haystack import Document
from haystack.components.preprocessors.csv_document_splitter import CSVDocumentSplitter


@pytest.fixture
def splitter() -> CSVDocumentSplitter:
    return CSVDocumentSplitter()


@pytest.fixture
def two_tables_sep_by_two_empty_rows() -> str:
    return """A,B,C
1,2,3
,,
,,
X,Y,Z
7,8,9
"""


@pytest.fixture
def two_tables_sep_by_two_empty_columns() -> str:
    return """A,B,,,X,Y
1,2,,,7,8
3,4,,,9,10
"""


class TestFindSplitIndices:
    def test_find_split_indices_row_two_tables(
        self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_rows: str
    ) -> None:
        df = pd.read_csv(StringIO(two_tables_sep_by_two_empty_rows), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="row")
        assert result == [(2, 3)]

    def test_find_split_indices_row_two_tables_with_empty_row(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,C
,,
1,2,3
,,
,,
X,Y,Z
7,8,9
"""
        df = pd.read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="row")
        assert result == [(3, 4)]

    def test_find_split_indices_row_three_tables(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,C
1,2,3
,,
,,
X,Y,Z
7,8,9
,,
,,
P,Q,R
"""
        df = pd.read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="row")
        assert result == [(2, 3), (6, 7)]

    def test_find_split_indices_column_two_tables(
        self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_columns: str
    ) -> None:
        df = pd.read_csv(StringIO(two_tables_sep_by_two_empty_columns), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=1, axis="column")
        assert result == [(2, 3)]

    def test_find_split_indices_column_two_tables_with_empty_column(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,,B,,,X,Y
1,,2,,,7,8
3,,4,,,9,10
"""
        df = pd.read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="column")
        assert result == [(3, 4)]

    def test_find_split_indices_column_three_tables(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,,,X,Y,,,P,Q
1,2,,,7,8,,,11,12
3,4,,,9,10,,,13,14
"""
        df = pd.read_csv(StringIO(csv_content), header=None, dtype=object)  # type: ignore
        result = splitter._find_split_indices(df, split_threshold=2, axis="column")
        assert result == [(2, 3), (6, 7)]


class TestCSVDocumentSplitter:
    def test_single_table_no_split(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,C
1,2,3
4,5,6
"""
        doc = Document(content=csv_content)
        result = splitter.run([doc])["documents"]
        assert len(result) == 1
        assert result[0].content == csv_content

    def test_row_split(self, splitter: CSVDocumentSplitter, two_tables_sep_by_two_empty_rows: str) -> None:
        doc = Document(content=two_tables_sep_by_two_empty_rows)
        result = splitter.run([doc])["documents"]
        assert len(result) == 2
        expected_tables = ["A,B,C\n1,2,3\n", "X,Y,Z\n7,8,9\n"]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]

    def test_column_split(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,,,X,Y
1,2,,,7,8
3,4,,,9,10
"""
        doc = Document(content=csv_content)
        result = splitter.run([doc])["documents"]
        assert len(result) == 2
        expected_tables = ["A,B\n1,2\n3,4\n", "X,Y\n7,8\n9,10\n"]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]

    def test_recursive_split(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = """A,B,,,X,Y
1,2,,,7,8
,,,,,
,,,,,
P,Q,,,M,N
3,4,,,9,10
"""
        doc = Document(content=csv_content)
        result = splitter.run([doc])["documents"]
        assert len(result) == 4
        expected_tables = ["A,B\n1,2\n", "X,Y\n7,8\n", "P,Q\n3,4\n", "M,N\n9,10\n"]
        for i, table in enumerate(result):
            assert table.content == expected_tables[i]

    def test_threshold_no_effect(self, two_tables_sep_by_two_empty_rows: str) -> None:
        splitter = CSVDocumentSplitter(row_split_threshold=3)
        doc = Document(content=two_tables_sep_by_two_empty_rows)
        result = splitter.run([doc])["documents"]
        assert len(result) == 1

    def test_empty_input(self, splitter: CSVDocumentSplitter) -> None:
        csv_content = ""
        doc = Document(content=csv_content)
        result = splitter.run([doc])["documents"]
        assert len(result) == 1
        assert result[0].content == csv_content

    def test_empty_documents(self, splitter: CSVDocumentSplitter) -> None:
        result = splitter.run([])["documents"]
        assert len(result) == 0
