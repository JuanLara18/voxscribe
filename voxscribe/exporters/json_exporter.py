"""JSON exporter — raw structured transcript data."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from voxscribe.models import MergedSegment

logger = logging.getLogger(__name__)


class JSONExporter:
    """Exports transcript segments as a JSON array.

    Each element in the array corresponds to one :class:`~voxscribe.models.MergedSegment`.
    The schema is stable and intended for programmatic consumption by other tools.

    Output schema::

        [
          {
            "speaker":    "SPEAKER_00",
            "start":      12.34,
            "end":        15.67,
            "text":       "Hello everyone.",
            "words": [
              {"word": "Hello", "start": 12.34, "end": 12.80, "confidence": 0.99},
              ...
            ]
          },
          ...
        ]

    ``words`` is ``null`` when the transcription backend did not produce
    word-level timestamps (e.g. plain faster-whisper without WhisperX alignment).
    """

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

        data = [dataclasses.asdict(seg) for seg in segments]

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("JSON written → %s (%d segments)", output_path, len(segments))
