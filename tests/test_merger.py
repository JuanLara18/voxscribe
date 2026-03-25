"""Unit tests for voxscribe.alignment.merger."""

from __future__ import annotations

import pytest

from voxscribe.alignment.merger import SegmentMerger, _overlap
from voxscribe.models import DiarizationSegment, TranscriptSegment


class TestOverlapHelper:
    def test_full_overlap(self):
        assert _overlap(0.0, 5.0, 0.0, 5.0) == pytest.approx(5.0)

    def test_partial_overlap(self):
        assert _overlap(0.0, 3.0, 2.0, 5.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _overlap(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)

    def test_contained(self):
        assert _overlap(0.0, 10.0, 2.0, 4.0) == pytest.approx(2.0)


class TestSegmentMerger:
    def test_basic_overlap_merge(self, sample_transcript_segments, sample_diarization_segments):
        merger = SegmentMerger()
        merged = merger.merge(sample_transcript_segments, sample_diarization_segments)

        assert len(merged) == len(sample_transcript_segments)
        # All segments should have a speaker assigned
        for seg in merged:
            assert seg.speaker in ("SPEAKER_00", "SPEAKER_01")

    def test_speaker_00_for_early_segments(self, sample_transcript_segments, sample_diarization_segments):
        merger = SegmentMerger()
        merged = merger.merge(sample_transcript_segments, sample_diarization_segments)
        merged.sort(key=lambda s: s.start)

        # Segments 0-1 fall in SPEAKER_00's range (0.0–5.5)
        assert merged[0].speaker == "SPEAKER_00"
        assert merged[1].speaker == "SPEAKER_00"

    def test_speaker_01_for_late_segments(self, sample_transcript_segments, sample_diarization_segments):
        merger = SegmentMerger()
        merged = merger.merge(sample_transcript_segments, sample_diarization_segments)
        merged.sort(key=lambda s: s.start)

        # Segments 2-3 fall in SPEAKER_01's range (5.5–11.0)
        assert merged[2].speaker == "SPEAKER_01"
        assert merged[3].speaker == "SPEAKER_01"

    def test_empty_transcript_raises(self, sample_diarization_segments):
        merger = SegmentMerger()
        with pytest.raises(ValueError, match="empty"):
            merger.merge([], sample_diarization_segments)

    def test_empty_diarization_raises(self, sample_transcript_segments):
        merger = SegmentMerger()
        with pytest.raises(ValueError, match="empty"):
            merger.merge(sample_transcript_segments, [])

    def test_result_sorted_by_time(self, sample_transcript_segments, sample_diarization_segments):
        merger = SegmentMerger()
        merged = merger.merge(sample_transcript_segments, sample_diarization_segments)
        starts = [s.start for s in merged]
        assert starts == sorted(starts)

    def test_text_preserved(self, sample_transcript_segments, sample_diarization_segments):
        merger = SegmentMerger()
        merged = merger.merge(sample_transcript_segments, sample_diarization_segments)
        texts = {s.text for s in merged}
        assert "Hello everyone." in texts

    def test_proximity_match(self):
        """Segments that don't overlap should still get assigned via proximity."""
        transcript = [TranscriptSegment(start=3.0, end=4.0, text="Nearby")]
        diarization = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0)]
        merger = SegmentMerger(max_gap=2.0)  # generous gap
        merged = merger.merge(transcript, diarization)
        assert merged[0].speaker == "SPEAKER_00"
