"""VoxScribe configuration via Pydantic Settings.

Settings are resolved in this priority order (highest → lowest):
  1. Explicit keyword arguments when constructing ``VoxScribeConfig``
  2. Environment variables (with ``VOXSCRIBE_`` prefix, except ``HF_TOKEN``)
  3. ``.env`` file in the current working directory
  4. Default values defined on the model

Usage::

    # From CLI code (values already parsed by Typer):
    cfg = VoxScribeConfig(model="large-v3-turbo", hf_token="hf_abc...")

    # From environment / .env only:
    cfg = VoxScribeConfig()

    # From the library:
    from voxscribe import Transcriber
    result = Transcriber(model="large-v3-turbo").run("audio.mp3")
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoxScribeConfig(BaseSettings):
    """All runtime configuration for a VoxScribe pipeline run."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VOXSCRIBE_",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Transcription ─────────────────────────────────────────────────────

    model: str = Field(
        default="base",
        description=(
            "Whisper model size. Recommended: 'base' (fast, good), "
            "'large-v3-turbo' (SOTA speed/accuracy), 'large-v3' (maximum accuracy)."
        ),
    )
    backend: Literal["faster-whisper", "whisperx"] = Field(
        default="faster-whisper",
        description=(
            "'faster-whisper' for standard transcription (always available). "
            "'whisperx' for word-level timestamps and integrated diarization "
            "(requires `pip install voxscribe[alignment]`)."
        ),
    )
    language: str | None = Field(
        default=None,
        description="ISO-639-1 language code (e.g. 'en', 'es'). None = auto-detect.",
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="'auto' selects CUDA if available, otherwise CPU.",
    )
    compute_type: str = Field(
        default="int8",
        description=(
            "Quantization type for faster-whisper: 'int8' (fastest CPU), "
            "'float16' (fastest CUDA), 'float32' (highest precision). "
            "Use 'auto' to let VoxScribe choose based on device."
        ),
    )

    # ── Diarization ───────────────────────────────────────────────────────

    diarization: bool = Field(
        default=True,
        description="Enable speaker diarization. Disable with --no-diarization.",
    )
    hf_token: str | None = Field(
        default=None,
        # Accept both VOXSCRIBE_HF_TOKEN and the standard HF_TOKEN env var.
        validation_alias=AliasChoices("hf_token", "VOXSCRIBE_HF_TOKEN", "HF_TOKEN"),
        description=(
            "HuggingFace access token for pyannote diarization. "
            "Without this, VoxScribe falls back to the built-in MFCC diarizer. "
            "Get yours at https://huggingface.co/settings/tokens"
        ),
    )
    min_speakers: int | None = Field(
        default=None,
        description="Minimum number of expected speakers (optional hint).",
    )
    max_speakers: int | None = Field(
        default=None,
        description="Maximum number of expected speakers (optional hint).",
    )

    # ── Output ────────────────────────────────────────────────────────────

    output_dir: str = Field(
        default="output",
        description="Directory where output files are written.",
    )
    formats: list[str] = Field(
        default=["md", "json"],
        description="Output formats: 'md', 'json', 'srt', 'vtt', 'txt'.",
    )
    title: str | None = Field(
        default=None,
        description="Document title (used as filename stem and Markdown heading).",
    )

    # ── Summarization ─────────────────────────────────────────────────────

    summarize: bool = Field(
        default=False,
        description="Generate an LLM summary via Ollama (requires `pip install voxscribe[summarization]`).",
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model name to use for summarization.",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API host URL.",
    )

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("formats", mode="before")
    @classmethod
    def _parse_formats(cls, v: list | str) -> list[str]:
        """Allow comma-separated string from env vars: VOXSCRIBE_FORMATS=md,json,srt."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, v: list[str]) -> list[str]:
        valid = {"md", "json", "srt", "vtt", "txt"}
        unknown = set(v) - valid
        if unknown:
            raise ValueError(f"Unknown format(s): {unknown}. Valid: {valid}")
        return v

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        # Only validate against known sizes when using faster-whisper.
        # WhisperX and custom HuggingFace repos may use other identifiers.
        known = {
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "medium", "medium.en",
            "large-v1", "large-v2", "large-v3",
            "large-v3-turbo",
            "distil-large-v3", "distil-medium.en", "distil-small.en",
        }
        if v not in known:
            # Allow custom model paths/repo IDs — just warn, don't raise.
            import logging
            logging.getLogger(__name__).warning(
                "Model '%s' is not in the list of known faster-whisper sizes. "
                "Proceeding — make sure the model identifier is correct.",
                v,
            )
        return v
