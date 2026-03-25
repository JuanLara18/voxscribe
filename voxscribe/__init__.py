"""VoxScribe — local, privacy-preserving transcription and speaker diarization.

Transcribes any audio or video file using state-of-the-art local AI models.
No data leaves your machine.

Quick start::

    from voxscribe import Transcriber

    # Basic usage (no token required)
    result = Transcriber(model="base").run("lecture.mp4")
    result.save("output/", formats=["md", "txt"])

    # With pyannote diarization (best accuracy)
    result = Transcriber(
        model="large-v3-turbo",
        hf_token="hf_...",          # get at huggingface.co/settings/tokens
    ).run("interview.mp4")
    result.save("output/", formats=["md", "srt", "json"], title="interview")

    # Word-level timestamps via WhisperX
    result = Transcriber(
        model="large-v3-turbo",
        backend="whisperx",
        hf_token="hf_...",
    ).run("podcast.mp3")

    # With Ollama summarization
    result = Transcriber(model="large-v3-turbo", summarize=True).run("meeting.mp4")
    print(result.summary)

CLI::

    voxscribe meeting.mp4 --model large-v3-turbo --hf-token $HF_TOKEN -f srt -f md
    voxscribe podcast.mp3 --no-diarization -f txt
    voxscribe interview.mp4 --model large-v3-turbo --summarize
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "VoxScribe Contributors"
__license__ = "MIT"

from voxscribe.config import VoxScribeConfig
from voxscribe.models import (
    DiarizationSegment,
    MergedSegment,
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
)


class Transcriber:
    """High-level façade for running the VoxScribe pipeline programmatically.

    This class provides a simple, keyword-argument-driven interface over
    :class:`~voxscribe.pipeline.Pipeline` and
    :class:`~voxscribe.config.VoxScribeConfig`.

    All constructor arguments map directly to :class:`VoxScribeConfig` fields.
    See that class for full documentation of each option.

    Example::

        from voxscribe import Transcriber

        result = Transcriber(
            model="large-v3-turbo",
            hf_token="hf_...",
            formats=["md", "srt"],
        ).run("podcast.mp3")

        print(f"Detected language: {result.language}")
        print(f"Speakers: {result.speakers}")
        print(result.text[:200])
    """

    def __init__(self, **kwargs) -> None:
        """
        Args:
            **kwargs: Any field from :class:`VoxScribeConfig`.
                Common options:

                - ``model`` (str): Whisper model size. Default: ``'base'``.
                - ``backend`` (str): ``'faster-whisper'`` or ``'whisperx'``.
                - ``language`` (str | None): Force language code.
                - ``device`` (str): ``'auto'``, ``'cpu'``, or ``'cuda'``.
                - ``diarization`` (bool): Enable speaker diarization.
                - ``hf_token`` (str | None): HuggingFace token for pyannote.
                - ``min_speakers`` / ``max_speakers`` (int | None): Speaker bounds.
                - ``output_dir`` (str): Output directory.
                - ``formats`` (list[str]): Output formats.
                - ``summarize`` (bool): Enable Ollama summarization.
                - ``ollama_model`` (str): Ollama model name.
        """
        self.config = VoxScribeConfig(**kwargs)

    def run(self, input_path: str) -> TranscriptResult:
        """Run the full pipeline on *input_path*.

        Args:
            input_path: Path to any audio or video file.

        Returns:
            :class:`~voxscribe.models.TranscriptResult` with all segments,
            detected language, duration, and optional summary.
        """
        from voxscribe.pipeline import Pipeline  # noqa: PLC0415

        return Pipeline(self.config).run(input_path)


__all__ = [
    "__version__",
    "Transcriber",
    "VoxScribeConfig",
    "TranscriptResult",
    "MergedSegment",
    "TranscriptSegment",
    "DiarizationSegment",
    "WordTimestamp",
]
