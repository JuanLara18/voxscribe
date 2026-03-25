"""Abstract protocol for transcription backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from voxscribe.models import TranscriptSegment


@runtime_checkable
class BaseTranscriber(Protocol):
    """Structural protocol that all transcription backends must satisfy.

    Implementations do not need to inherit from this class — they only need
    to provide a ``transcribe`` method with the correct signature.
    """

    def transcribe(self, audio_path: Path) -> tuple[list[TranscriptSegment], str | None]:
        """Transcribe an audio file.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.

        Returns:
            A tuple of:
            - List of :class:`~voxscribe.models.TranscriptSegment` objects
              sorted by start time.
            - Detected (or forced) language code string (e.g. ``'en'``),
              or ``None`` if unknown.
        """
        ...
