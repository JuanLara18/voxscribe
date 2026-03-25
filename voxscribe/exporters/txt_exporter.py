"""Plain-text exporter.

Produces a clean, human-readable transcript without any markup.
Ideal for feeding into downstream LLM pipelines, copy-paste, or simple
archiving.

Format::

    [00:00:04] SPEAKER_00: Hello and welcome to the show.
    [00:00:07] SPEAKER_01: Thanks for having me.

Speaker labels and timestamps are optional and controlled by constructor flags.
"""

from __future__ import annotations

import logging
from pathlib import Path

from voxscribe._utils import format_timestamp_hms
from voxscribe.models import MergedSegment

logger = logging.getLogger(__name__)


class TXTExporter:
    """Exports transcript as plain text (.txt)."""

    def __init__(
        self,
        include_timestamps: bool = True,
        include_speakers: bool = True,
    ) -> None:
        """
        Args:
            include_timestamps: Prepend ``[HH:MM:SS]`` to each line.
            include_speakers: Prepend speaker label to each line.
        """
        self.include_timestamps = include_timestamps
        self.include_speakers = include_speakers

    def export(
        self,
        segments: list[MergedSegment],
        output_path: Path,
        *,
        title: str | None = None,
        summary: str | None = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        unique_speakers = {s.speaker for s in segments}
        show_speaker = self.include_speakers and len(unique_speakers) > 1

        with output_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                parts: list[str] = []

                if self.include_timestamps:
                    parts.append(f"[{format_timestamp_hms(seg.start)}]")

                if show_speaker:
                    parts.append(f"{seg.speaker}:")

                parts.append(seg.text.strip())
                f.write(" ".join(parts) + "\n")

            if summary:
                f.write("\n\n--- SUMMARY ---\n\n")
                f.write(summary.strip())
                f.write("\n")

        logger.info("TXT written → %s (%d lines)", output_path, len(segments))
