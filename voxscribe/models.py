"""Core data models shared across all VoxScribe components.

These dataclasses define the internal data contract between pipeline stages.
All components receive and return instances of these types — never raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING


# ── Atomic building blocks ────────────────────────────────────────────────────


@dataclass
class WordTimestamp:
    """A single word with precise start/end timing and optional confidence.

    Populated only when using WhisperX (forced alignment backend).
    """

    word: str
    start: float
    end: float
    confidence: float = 1.0


# ── Pipeline stage outputs ────────────────────────────────────────────────────


@dataclass
class TranscriptSegment:
    """One ASR segment produced by the transcription backend.

    Corresponds to a single Whisper chunk. Words are populated only when
    the backend supports forced word-level alignment (WhisperX).
    """

    start: float
    end: float
    text: str
    words: list[WordTimestamp] | None = None


@dataclass
class DiarizationSegment:
    """One speaker turn produced by the diarization backend."""

    speaker: str
    start: float
    end: float


@dataclass
class MergedSegment:
    """A transcript segment enriched with a speaker label.

    This is the canonical output of the pipeline — every consumer
    (exporter, summarizer, etc.) works with lists of MergedSegment.
    """

    speaker: str
    start: float
    end: float
    text: str
    words: list[WordTimestamp] | None = None


# ── Final result container ────────────────────────────────────────────────────


@dataclass
class TranscriptResult:
    """Complete output of a VoxScribe pipeline run.

    Returned by :class:`~voxscribe.Transcriber` and
    :class:`~voxscribe.pipeline.Pipeline`. Contains all merged segments plus
    metadata and an optional LLM-generated summary.
    """

    segments: list[MergedSegment]
    language: str | None = None
    duration: float | None = None
    summary: str | None = None

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def text(self) -> str:
        """Full transcript as a single plain-text string."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def speakers(self) -> list[str]:
        """Unique speaker labels in order of first appearance."""
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen

    # ── Save shortcut ─────────────────────────────────────────────────────

    def save(
        self,
        output_dir: str | Path,
        formats: list[str] | None = None,
        title: str | None = None,
    ) -> dict[str, Path]:
        """Write the transcript to one or more output formats.

        This is a convenience wrapper around the exporter classes. For
        fine-grained control, use the individual exporters directly.

        Args:
            output_dir: Directory to write output files into.
            formats: Formats to write.  Supported values: ``"md"``,
                ``"json"``, ``"srt"``, ``"vtt"``, ``"txt"``.
                Defaults to ``["md", "json"]``.
            title: Document title used as the filename stem and the
                Markdown heading.  Defaults to ``"transcript"``.

        Returns:
            Mapping from format name to the written :class:`~pathlib.Path`.

        Example::

            result = Transcriber(model="large-v3-turbo").run("interview.mp4")
            paths = result.save("output/", formats=["md", "srt"], title="interview")
        """
        # Lazy import to avoid circular dependency (models ← exporters → models).
        from voxscribe.exporters import get_exporter  # noqa: PLC0415

        if formats is None:
            formats = ["md", "json"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = title or "transcript"
        paths: dict[str, Path] = {}

        for fmt in formats:
            exporter = get_exporter(fmt)
            ext = "md" if fmt == "md" else fmt
            out_path = output_dir / f"{stem}.{ext}"
            exporter.export(self.segments, out_path, title=title, summary=self.summary)
            paths[fmt] = out_path

        return paths
