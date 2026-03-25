"""WebVTT subtitle exporter.

WebVTT is the web standard for subtitles (HTML5 ``<track>`` element).
Differences from SRT: ``WEBVTT`` header, ``.`` instead of ``,`` in timestamps,
and support for ``<v Speaker>`` voice tags for rich player rendering.

Format::

    WEBVTT

    00:00:12.340 --> 00:00:15.670
    <v SPEAKER_00>Hello everyone.</v>

    00:00:15.670 --> 00:00:18.900
    <v SPEAKER_01>Thanks for having me.</v>
"""

from __future__ import annotations

import logging
from pathlib import Path

from voxscribe._utils import format_timestamp
from voxscribe.models import MergedSegment

logger = logging.getLogger(__name__)


class VTTExporter:
    """Exports transcript as WebVTT (.vtt) subtitle file."""

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
            f.write("WEBVTT\n\n")

            for seg in segments:
                start_ts = format_timestamp(seg.start, ms_separator=".")
                end_ts = format_timestamp(seg.end, ms_separator=".")
                text = seg.text.strip()

                f.write(f"{start_ts} --> {end_ts}\n")
                if show_speaker:
                    f.write(f"<v {seg.speaker}>{text}</v>\n\n")
                else:
                    f.write(f"{text}\n\n")

        logger.info("VTT written → %s (%d cues)", output_path, len(segments))
