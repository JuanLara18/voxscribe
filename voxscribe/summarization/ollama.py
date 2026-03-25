"""Local LLM summarization via Ollama.

Ollama runs large language models (Llama 3.2, Mistral, Qwen 3, …) locally
with a simple REST API.  VoxScribe uses it to generate structured meeting
summaries without sending any data to external services.

Install Ollama: https://ollama.com/
Pull a model:   ollama pull llama3.2
Install SDK:    pip install "voxscribe[summarization]"

The summarizer is entirely optional — the pipeline skips it gracefully when
``--summarize`` is not passed or when Ollama is unreachable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxscribe.models import MergedSegment

from voxscribe._utils import format_timestamp_hms

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """\
You are an expert note-taker. Analyze the following transcript and provide a
concise, structured summary in Markdown format with these sections:

## Overview
(2-3 sentence summary of the main topic and outcome)

## Key Topics
(bullet list of main subjects discussed)

## Decisions
(bullet list of decisions made, or "None identified" if none)

## Action Items
(bullet list with owner if mentioned, or "None identified" if none)

---

Transcript:
{transcript}
"""


class OllamaSummarizer:
    """Generates structured summaries from transcript segments using Ollama.

    Example::

        summarizer = OllamaSummarizer(model="llama3.2")
        summary = summarizer.summarize(merged_segments)
        print(summary)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        prompt_template: str | None = None,
    ) -> None:
        """
        Args:
            model: Ollama model name (must be pulled via ``ollama pull <model>``).
            host: Ollama API base URL.
            prompt_template: Custom prompt template.  Use ``{transcript}`` as
                the placeholder for the transcript text.  Defaults to the
                built-in structured summary prompt.
        """
        self.model = model
        self.host = host
        self.prompt_template = prompt_template or _DEFAULT_PROMPT

    # ── Public API ────────────────────────────────────────────────────────

    def summarize(self, segments: list[MergedSegment]) -> str:
        """Generate a summary of *segments* using the configured Ollama model.

        Args:
            segments: Merged transcript segments.

        Returns:
            Markdown-formatted summary string.

        Raises:
            ImportError: If the ``ollama`` package is not installed.
            RuntimeError: If Ollama is not running or the request fails.
        """
        try:
            import ollama  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The ollama package is not installed. "
                "Run: pip install 'voxscribe[summarization]'"
            ) from exc

        transcript_text = self._format_transcript(segments)
        prompt = self.prompt_template.format(transcript=transcript_text)

        logger.info("Requesting summary from Ollama model '%s' …", self.model)
        try:
            client = ollama.Client(host=self.host)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            summary: str = response.message.content
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}\n"
                "Make sure Ollama is running (ollama serve) and "
                f"the model '{self.model}' is pulled (ollama pull {self.model})."
            ) from exc

        logger.info("Summary generated (%d chars).", len(summary))
        return summary

    def is_available(self) -> bool:
        """Return ``True`` if Ollama is reachable and the model is available."""
        try:
            import ollama  # noqa: PLC0415

            client = ollama.Client(host=self.host)
            models = client.list()
            names = [m.model for m in models.models]
            return any(self.model in n for n in names)
        except Exception:  # noqa: BLE001
            return False

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_transcript(segments: list[MergedSegment]) -> str:
        lines = [
            f"{seg.speaker} [{format_timestamp_hms(seg.start)}]: {seg.text}"
            for seg in segments
        ]
        return "\n".join(lines)
