# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.converters.msg import MSGToDocument


class TestMSGToDocument:
    def test_run(self, test_files_path):
        converter = MSGToDocument(store_full_path=True)
        paths = [test_files_path / "msg" / "sample.msg"]
        result = converter.run(sources=paths, meta={"date_added": "2021-09-01T00:00:00"})
        assert len(result["documents"]) == 1
        assert result["documents"][0].content.startswith('From: "Sebastian Lee"')
        assert result["documents"][0].meta == {
            "date_added": "2021-09-01T00:00:00",
            "file_path": str(test_files_path / "msg" / "sample.msg"),
        }
        assert len(result["attachments"]) == 1
        assert result["attachments"][0].mime_type == "application/pdf"
        assert result["attachments"][0].meta == {
            "date_added": "2021-09-01T00:00:00",
            "parent_file_path": str(test_files_path / "msg" / "sample.msg"),
            "file_path": "sample_pdf_1.pdf",
        }

    def test_run_wrong_file_type(self, test_files_path, caplog):
        converter = MSGToDocument(store_full_path=False)
        paths = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        result = converter.run(sources=paths, meta={"date_added": "2021-09-01T00:00:00"})
        assert len(result["documents"]) == 0
        assert "msg_file is not an Outlook MSG file" in caplog.text

    def test_run_empty_sources(self, test_files_path):
        converter = MSGToDocument(store_full_path=False)
        result = converter.run(sources=[])
        assert len(result["documents"]) == 0
        assert len(result["attachments"]) == 0
