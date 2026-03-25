"""Abstract protocol for speaker diarization backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from voxscribe.models import DiarizationSegment


@runtime_checkable
class BaseDiarizer(Protocol):
    """Structural protocol that all diarization backends must satisfy."""

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Identify speaker turns in *audio_path*.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.
            min_speakers: Optional lower bound on speaker count.
            max_speakers: Optional upper bound on speaker count.

        Returns:
            List of :class:`~voxscribe.models.DiarizationSegment` objects
            sorted by start time.
        """
        ...
