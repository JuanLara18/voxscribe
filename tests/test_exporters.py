"""Unit tests for all output format exporters."""

from __future__ import annotations

import json

import pytest

from voxscribe.exporters import get_exporter
from voxscribe.exporters.json_exporter import JSONExporter
from voxscribe.exporters.markdown_exporter import MarkdownExporter
from voxscribe.exporters.srt_exporter import SRTExporter
from voxscribe.exporters.txt_exporter import TXTExporter
from voxscribe.exporters.vtt_exporter import VTTExporter


class TestGetExporter:
    def test_known_formats(self):
        for fmt in ["md", "json", "srt", "vtt", "txt"]:
            exporter = get_exporter(fmt)
            assert exporter is not None

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown output format"):
            get_exporter("pdf")

    def test_case_insensitive(self):
        assert get_exporter("JSON") is not None
        assert get_exporter("SRT") is not None


class TestJSONExporter:
    def test_output_is_valid_json(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.json"
        JSONExporter().export(sample_merged_segments, path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == len(sample_merged_segments)

    def test_segment_fields(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.json"
        JSONExporter().export(sample_merged_segments, path)
        data = json.loads(path.read_text())
        first = data[0]
        assert "speaker" in first
        assert "start" in first
        assert "end" in first
        assert "text" in first


class TestMarkdownExporter:
    def test_creates_file(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.md"
        MarkdownExporter().export(sample_merged_segments, path)
        assert path.exists()

    def test_contains_speaker_names(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.md"
        MarkdownExporter().export(sample_merged_segments, path)
        content = path.read_text()
        assert "**SPEAKER_00**" in content
        assert "**SPEAKER_01**" in content

    def test_contains_text(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.md"
        MarkdownExporter().export(sample_merged_segments, path)
        content = path.read_text()
        assert "Hello everyone." in content

    def test_custom_title(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.md"
        MarkdownExporter().export(sample_merged_segments, path, title="My Interview")
        content = path.read_text()
        assert "# My Interview" in content

    def test_summary_appended(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.md"
        MarkdownExporter().export(
            sample_merged_segments, path, summary="## Overview\nGreat discussion."
        )
        content = path.read_text()
        assert "## Summary" in content
        assert "Great discussion." in content


class TestSRTExporter:
    def test_creates_file(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.srt"
        SRTExporter().export(sample_merged_segments, path)
        assert path.exists()

    def test_srt_format(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.srt"
        SRTExporter().export(sample_merged_segments, path)
        content = path.read_text()
        # Should start with index 1
        assert content.startswith("1\n")
        # Should contain SRT timestamp format (comma separator)
        assert "," in content
        assert "-->" in content

    def test_speaker_labels_in_srt(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.srt"
        SRTExporter().export(sample_merged_segments, path)
        content = path.read_text()
        assert "SPEAKER_00:" in content


class TestVTTExporter:
    def test_creates_file(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.vtt"
        VTTExporter().export(sample_merged_segments, path)
        assert path.exists()

    def test_vtt_header(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.vtt"
        VTTExporter().export(sample_merged_segments, path)
        content = path.read_text()
        assert content.startswith("WEBVTT\n")

    def test_vtt_dot_separator(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.vtt"
        VTTExporter().export(sample_merged_segments, path)
        content = path.read_text()
        # VTT uses '.' not ',' in timestamps
        assert "00:00:00.000 -->" in content

    def test_voice_tags(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.vtt"
        VTTExporter().export(sample_merged_segments, path)
        content = path.read_text()
        assert "<v SPEAKER_00>" in content


class TestTXTExporter:
    def test_creates_file(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.txt"
        TXTExporter().export(sample_merged_segments, path)
        assert path.exists()

    def test_one_line_per_segment(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.txt"
        TXTExporter().export(sample_merged_segments, path)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(sample_merged_segments)

    def test_no_timestamps(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.txt"
        TXTExporter(include_timestamps=False).export(sample_merged_segments, path)
        content = path.read_text()
        assert "[" not in content

    def test_summary_appended(self, sample_merged_segments, tmp_path):
        path = tmp_path / "out.txt"
        TXTExporter().export(sample_merged_segments, path, summary="Action item: deploy")
        content = path.read_text()
        assert "--- SUMMARY ---" in content
        assert "Action item: deploy" in content
