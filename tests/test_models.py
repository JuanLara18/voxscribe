"""Unit tests for voxscribe.models."""

from __future__ import annotations

import pytest

from voxscribe.models import (
    DiarizationSegment,
    MergedSegment,
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
)


class TestWordTimestamp:
    def test_defaults(self):
        w = WordTimestamp(word="hello", start=0.0, end=0.5)
        assert w.confidence == 1.0

    def test_custom_confidence(self):
        w = WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.92)
        assert w.confidence == 0.92


class TestTranscriptSegment:
    def test_basic(self):
        seg = TranscriptSegment(start=1.0, end=3.5, text="Test")
        assert seg.words is None

    def test_with_words(self):
        words = [WordTimestamp("Test", 1.0, 1.3)]
        seg = TranscriptSegment(start=1.0, end=3.5, text="Test", words=words)
        assert seg.words is not None
        assert len(seg.words) == 1


class TestTranscriptResult:
    def test_text_property(self, sample_merged_segments):
        result = TranscriptResult(segments=sample_merged_segments)
        assert "Hello everyone" in result.text
        assert "Let's get started" in result.text

    def test_speakers_property(self, sample_merged_segments):
        result = TranscriptResult(segments=sample_merged_segments)
        speakers = result.speakers
        assert "SPEAKER_00" in speakers
        assert "SPEAKER_01" in speakers
        # Order of first appearance
        assert speakers[0] == "SPEAKER_00"
        assert speakers[1] == "SPEAKER_01"

    def test_save_creates_files(self, sample_merged_segments, tmp_path):
        result = TranscriptResult(segments=sample_merged_segments, language="en")
        paths = result.save(tmp_path, formats=["json", "txt"], title="test")
        assert paths["json"].exists()
        assert paths["txt"].exists()
        assert paths["json"].name == "test.json"
        assert paths["txt"].name == "test.txt"

    def test_save_all_formats(self, sample_merged_segments, tmp_path):
        result = TranscriptResult(segments=sample_merged_segments)
        paths = result.save(tmp_path, formats=["md", "json", "srt", "vtt", "txt"])
        for fmt in ["md", "json", "srt", "vtt", "txt"]:
            assert fmt in paths
            assert paths[fmt].exists()

    def test_save_unknown_format_raises(self, sample_merged_segments, tmp_path):
        result = TranscriptResult(segments=sample_merged_segments)
        with pytest.raises(ValueError, match="Unknown output format"):
            result.save(tmp_path, formats=["xyz"])
