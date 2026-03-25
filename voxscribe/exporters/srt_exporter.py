"""SRT (SubRip) subtitle exporter.

SRT is the most widely supported subtitle format, compatible with VLC,
YouTube, FFmpeg, and virtually every video player and editor.

Format::

    1
    00:00:12,340 --> 00:00:15,670
    SPEAKER_00: Hello everyone.

    2
    00:00:15,670 --> 00:00:18,900
    SPEAKER_01: Thanks for having me.

Speaker labels are prepended to the text.  When all segments share the same
speaker (or diarization was disabled), the label is omitted for cleaner output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from voxscribe._utils import format_timestamp
from voxscribe.models import MergedSegment

logger = logging.getLogger(__name__)


class SRTExporter:
    """Exports transcript as SubRip (.srt) subtitle file."""

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
        show_speaker = len(unique_speakers) > 1

        with output_path.open("w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                start_ts = format_timestamp(seg.start, ms_separator=",")
                end_ts = format_timestamp(seg.end, ms_separator=",")
                text = seg.text.strip()
                if show_speaker:
                    text = f"{seg.speaker}: {text}"

                f.write(f"{idx}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{text}\n\n")

        logger.info("SRT written → %s (%d cues)", output_path, len(segments))
