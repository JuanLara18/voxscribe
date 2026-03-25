"""Transcription backends subpackage."""

from __future__ import annotations

from voxscribe.transcription.base import BaseTranscriber


def get_transcriber(
    backend: str,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
    language: str | None = None,
    hf_token: str | None = None,
) -> BaseTranscriber:
    """Factory that returns the appropriate transcriber for *backend*.

    Args:
        backend: ``'faster-whisper'`` or ``'whisperx'``.
        model_size: Whisper model identifier.
        device: ``'auto'``, ``'cpu'``, or ``'cuda'``.
        compute_type: Quantization level.
        language: Forced language code or ``None``.
        hf_token: HuggingFace token (used by WhisperX diarization).

    Returns:
        An object satisfying the :class:`BaseTranscriber` protocol.

    Raises:
        ValueError: For an unknown backend name.
    """
    if backend == "faster-whisper":
        from voxscribe.transcription.faster_whisper import FasterWhisperTranscriber  # noqa

        return FasterWhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    if backend == "whisperx":
        from voxscribe.transcription.whisperx import WhisperXTranscriber  # noqa

        return WhisperXTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            hf_token=hf_token,
        )
    raise ValueError(
        f"Unknown transcription backend: '{backend}'. "
        "Valid options: 'faster-whisper', 'whisperx'."
    )


__all__ = ["BaseTranscriber", "get_transcriber"]
