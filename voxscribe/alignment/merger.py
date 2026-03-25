"""Aligns ASR transcript segments with speaker diarization turns.

Given two independent time-series (ASR output and diarization output),
this module produces a unified list of :class:`~voxscribe.models.MergedSegment`
objects — each carrying the original text plus the best matching speaker label.

Matching strategy (in order of preference):
  1. **Overlap** — assign the speaker whose turn overlaps most with the
     transcript segment (overlap ≥ ``min_overlap``).
  2. **Proximity** — if no overlap exceeds the threshold, assign the nearest
     speaker turn whose midpoint is within ``max_gap`` seconds.
  3. **Context** — if still unmatched, use the speaker of the temporally
     closest already-assigned neighbour within ``max_gap``.
  4. **Fallback** — label as ``'SPEAKER_00'`` (instead of UNKNOWN) so that
     downstream exporters always have a valid speaker string.
"""

from __future__ import annotations

import logging
from pathlib import Path

from voxscribe.models import DiarizationSegment, MergedSegment, TranscriptSegment

logger = logging.getLogger(__name__)


class SegmentMerger:
    """Merges ASR segments and diarization segments into speaker-labelled entries.

    Example::

        merger = SegmentMerger()
        merged = merger.merge(transcript_segments, diarization_segments)
    """

    def __init__(
        self,
        max_gap: float = 0.5,
        min_overlap: float = 0.1,
    ) -> None:
        """
        Args:
            max_gap: Maximum gap in seconds between an ASR segment and the
                nearest speaker turn for proximity / context matching.
            min_overlap: Minimum overlap in seconds required to accept an
                overlap-based match.
        """
        self.max_gap = max_gap
        self.min_overlap = min_overlap

    # ── Public API ────────────────────────────────────────────────────────

    def merge(
        self,
        transcript_segments: list[TranscriptSegment],
        diarization_segments: list[DiarizationSegment],
    ) -> list[MergedSegment]:
        """Produce speaker-labelled merged segments.

        Args:
            transcript_segments: ASR output, sorted by start time.
            diarization_segments: Diarization output, sorted by start time.

        Returns:
            List of :class:`~voxscribe.models.MergedSegment` sorted by start time.

        Raises:
            ValueError: If either input list is empty.
        """
        if not transcript_segments:
            raise ValueError("transcript_segments is empty.")
        if not diarization_segments:
            raise ValueError("diarization_segments is empty.")

        logger.info(
            "Merging %d transcript segments with %d speaker turns …",
            len(transcript_segments),
            len(diarization_segments),
        )

        matched: list[MergedSegment] = []
        unmatched: list[MergedSegment] = []   # No speaker yet

        for tseg in transcript_segments:
            speaker = self._find_best_speaker(tseg, diarization_segments)
            mseg = MergedSegment(
                speaker=speaker or "",          # Placeholder; filled below
                start=tseg.start,
                end=tseg.end,
                text=tseg.text,
                words=tseg.words,
            )
            if speaker:
                matched.append(mseg)
            else:
                unmatched.append(mseg)

        # Second pass: context-based assignment for unmatched segments.
        if unmatched:
            self._assign_from_context(matched, unmatched)

        all_segments = sorted(matched + unmatched, key=lambda s: s.start)

        unknown = sum(1 for s in all_segments if not s.speaker)
        if unknown:
            # Assign a generic fallback label so exporters never see empty strings.
            for s in all_segments:
                if not s.speaker:
                    s.speaker = "SPEAKER_00"
            logger.warning(
                "%d / %d segments (%.0f%%) could not be assigned a speaker and "
                "were labelled SPEAKER_00.",
                unknown, len(all_segments), 100 * unknown / len(all_segments),
            )

        n_speakers = len({s.speaker for s in all_segments})
        logger.info(
            "Merge complete — %d segments, %d unique speakers.",
            len(all_segments), n_speakers,
        )
        return all_segments

    # ── Core matching logic ───────────────────────────────────────────────

    def _find_best_speaker(
        self,
        tseg: TranscriptSegment,
        diar: list[DiarizationSegment],
    ) -> str | None:
        """Return the best matching speaker via overlap or proximity."""
        best_speaker: str | None = None
        best_overlap: float = 0.0

        for dseg in diar:
            ov = _overlap(tseg.start, tseg.end, dseg.start, dseg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = dseg.speaker

        if best_overlap >= self.min_overlap:
            return best_speaker

        # Fallback: nearest midpoint within max_gap.
        return self._find_nearest_speaker(tseg, diar)

    def _find_nearest_speaker(
        self,
        tseg: TranscriptSegment,
        diar: list[DiarizationSegment],
    ) -> str | None:
        mid = (tseg.start + tseg.end) / 2.0
        best_speaker: str | None = None
        min_dist = float("inf")

        for dseg in diar:
            d = abs(mid - (dseg.start + dseg.end) / 2.0)
            if d < min_dist:
                min_dist = d
                best_speaker = dseg.speaker

        return best_speaker if min_dist <= self.max_gap else None

    def _assign_from_context(
        self,
        matched: list[MergedSegment],
        unmatched: list[MergedSegment],
    ) -> None:
        """Assign speakers to *unmatched* using adjacent *matched* segments.

        Modifies both lists in-place: successfully assigned segments are
        appended to *matched* and removed from *unmatched*.
        """
        if not matched:
            return

        matched.sort(key=lambda s: s.start)
        unmatched.sort(key=lambda s: s.start)

        to_remove: list[int] = []

        for i, useg in enumerate(unmatched):
            prev = next(
                (m for m in reversed(matched) if m.end <= useg.start), None
            )
            nxt = next(
                (m for m in matched if m.start >= useg.end), None
            )

            speaker = self._closer_context(useg, prev, nxt)
            if speaker:
                useg.speaker = speaker
                matched.append(useg)
                to_remove.append(i)

        for idx in reversed(to_remove):
            unmatched.pop(idx)


    def _closer_context(
        self,
        useg: MergedSegment,
        prev: MergedSegment | None,
        nxt: MergedSegment | None,
    ) -> str | None:
        """Return the speaker of the closer context segment."""
        gap_prev = (useg.start - prev.end) if prev else float("inf")
        gap_next = (nxt.start - useg.end) if nxt else float("inf")

        if gap_prev <= gap_next and gap_prev <= self.max_gap and prev:
            return prev.speaker
        if gap_next <= self.max_gap and nxt:
            return nxt.speaker
        return None


# ── Module-level helpers ──────────────────────────────────────────────────────


def _overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    """Return the duration of overlap between two time intervals (≥ 0)."""
    return max(0.0, min(e1, e2) - max(s1, s2))
