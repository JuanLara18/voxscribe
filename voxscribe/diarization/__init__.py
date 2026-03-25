"""Speaker diarization subpackage."""

from __future__ import annotations

import logging

from voxscribe.diarization.base import BaseDiarizer

logger = logging.getLogger(__name__)


def get_diarizer(
    hf_token: str | None,
    device: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> BaseDiarizer:
    """Return the best available diarizer given the current environment.

    Selection logic:
      1. **pyannote** — used when ``hf_token`` is provided *and*
         ``pyannote-audio`` is installed.  Best accuracy (~8% DER).
      2. **SimpleDiarizer** — always available fallback using
         MFCC + agglomerative clustering.  No token or extra install required.

    Args:
        hf_token: HuggingFace token.  ``None`` or empty → forces SimpleDiarizer.
        device: ``'auto'``, ``'cpu'``, or ``'cuda'``.
        min_speakers: Hint passed through to the chosen diarizer.
        max_speakers: Hint passed through to the chosen diarizer.

    Returns:
        An object satisfying the :class:`BaseDiarizer` protocol.
    """
    if hf_token:
        try:
            from voxscribe.diarization.pyannote import PyannoteDiarizer  # noqa

            logger.info("HF token found — using pyannote diarizer (SOTA accuracy).")
            return PyannoteDiarizer(hf_token=hf_token, device=device)
        except ImportError:
            logger.warning(
                "pyannote.audio not installed. "
                "Run `pip install 'voxscribe[diarization]'` for best results. "
                "Falling back to built-in MFCC diarizer."
            )

    from voxscribe.diarization.simple import SimpleDiarizer  # noqa

    logger.info("Using built-in MFCC diarizer (no HF token required).")
    return SimpleDiarizer(device=device)


__all__ = ["BaseDiarizer", "get_diarizer"]
