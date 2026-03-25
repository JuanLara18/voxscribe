"""Pipeline orchestrator — coordinates all processing stages.

This module is the heart of VoxScribe.  :class:`Pipeline` wires together
audio extraction, transcription, diarization, alignment, summarization, and
export in the correct order, with clear logging and graceful degradation when
optional components are unavailable.

Usage::

    from voxscribe.config import VoxScribeConfig
    from voxscribe.pipeline import Pipeline

    cfg = VoxScribeConfig(model="large-v3-turbo", hf_token="hf_…")
    result = Pipeline(cfg).run("interview.mp4")
    result.save("output/", formats=["md", "srt"])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from voxscribe.config import VoxScribeConfig
from voxscribe.models import MergedSegment, TranscriptResult

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """End-to-end VoxScribe pipeline.

    Steps:
      1. Audio extraction  (FFmpeg)
      2. Transcription     (faster-whisper or WhisperX)
      3. Diarization       (pyannote / SimpleDiarizer / WhisperX-integrated)
      4. Alignment         (SegmentMerger, skipped when WhisperX handles it)
      5. Summarization     (Ollama, optional)
    """

    def __init__(self, config: VoxScribeConfig) -> None:
        self.cfg = config

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, input_path: str | Path) -> TranscriptResult:
        """Execute the full pipeline on *input_path*.

        Args:
            input_path: Path to any audio or video file.

        Returns:
            :class:`~voxscribe.models.TranscriptResult` containing all
            segments, detected language, duration, and optional summary.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "audio.wav"

        t_start = time.perf_counter()
        console.rule(f"[bold blue]VoxScribe[/] — {input_path.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            # ── Step 1: Extract audio ─────────────────────────────────────
            task = progress.add_task("[cyan]Extracting audio …", total=None)
            audio_path = self._extract_audio(input_path, audio_path)
            progress.update(task, description="[green]✓ Audio extracted")
            progress.stop_task(task)

            # ── Step 2+3+4: Transcribe (and optionally diarize) ───────────
            if self.cfg.backend == "whisperx" and self.cfg.diarization and self.cfg.hf_token:
                task = progress.add_task(
                    "[cyan]Transcribing + aligning + diarizing (WhisperX) …", total=None
                )
                merged_segments, language = self._run_whisperx_full(audio_path)
                progress.update(task, description="[green]✓ WhisperX pipeline complete")
                progress.stop_task(task)

            else:
                # Transcription
                task = progress.add_task(
                    f"[cyan]Transcribing with {self.cfg.backend} …", total=None
                )
                transcript_segments, language = self._transcribe(audio_path)
                progress.update(task, description=f"[green]✓ Transcription complete ({len(transcript_segments)} segments)")
                progress.stop_task(task)

                # Diarization + Merge
                if self.cfg.diarization:
                    task = progress.add_task("[cyan]Diarizing speakers …", total=None)
                    diar_segments = self._diarize(audio_path)
                    progress.update(task, description=f"[green]✓ Diarization complete ({len({s.speaker for s in diar_segments})} speakers)")
                    progress.stop_task(task)

                    task = progress.add_task("[cyan]Aligning segments …", total=None)
                    merged_segments = self._merge(transcript_segments, diar_segments)
                    progress.update(task, description="[green]✓ Alignment complete")
                    progress.stop_task(task)
                else:
                    merged_segments = [
                        MergedSegment(
                            speaker="SPEAKER_00",
                            start=s.start,
                            end=s.end,
                            text=s.text,
                            words=s.words,
                        )
                        for s in transcript_segments
                    ]

            # ── Step 5: Summarize (optional) ──────────────────────────────
            summary: str | None = None
            if self.cfg.summarize:
                task = progress.add_task(
                    f"[cyan]Summarizing with {self.cfg.ollama_model} …", total=None
                )
                summary = self._summarize(merged_segments)
                progress.update(task, description="[green]✓ Summary generated")
                progress.stop_task(task)

        # ── Build result ──────────────────────────────────────────────────
        duration = merged_segments[-1].end if merged_segments else 0.0
        result = TranscriptResult(
            segments=merged_segments,
            language=language,
            duration=duration,
            summary=summary,
        )

        # ── Auto-save ─────────────────────────────────────────────────────
        stem = self.cfg.title or input_path.stem
        saved_paths = result.save(output_dir, formats=self.cfg.formats, title=stem)

        elapsed = time.perf_counter() - t_start
        console.rule("[bold green]Done")
        console.print(f"[dim]Total time: {elapsed:.1f}s[/]")
        for fmt, path in saved_paths.items():
            console.print(f"  [bold]{fmt.upper()}[/] → {path}")

        return result

    # ── Pipeline steps ────────────────────────────────────────────────────

    def _extract_audio(self, input_path: Path, audio_path: Path) -> Path:
        from voxscribe.audio import AudioExtractor  # noqa

        return AudioExtractor().extract(input_path, audio_path)

    def _transcribe(self, audio_path: Path):
        from voxscribe.transcription import get_transcriber  # noqa

        transcriber = get_transcriber(
            backend=self.cfg.backend,
            model_size=self.cfg.model,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            language=self.cfg.language,
            hf_token=self.cfg.hf_token,
        )
        return transcriber.transcribe(audio_path)

    def _run_whisperx_full(self, audio_path: Path):
        """WhisperX integrated path: transcription + alignment + diarization."""
        from voxscribe.transcription.whisperx import WhisperXTranscriber  # noqa

        t = WhisperXTranscriber(
            model_size=self.cfg.model,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            language=self.cfg.language,
            hf_token=self.cfg.hf_token,
        )
        return t.transcribe_with_diarization(
            audio_path,
            min_speakers=self.cfg.min_speakers,
            max_speakers=self.cfg.max_speakers,
        )

    def _diarize(self, audio_path: Path):
        from voxscribe.diarization import get_diarizer  # noqa

        diarizer = get_diarizer(
            hf_token=self.cfg.hf_token,
            device=self.cfg.device,
        )
        return diarizer.diarize(
            audio_path,
            min_speakers=self.cfg.min_speakers,
            max_speakers=self.cfg.max_speakers,
        )

    def _merge(self, transcript_segments, diar_segments):
        from voxscribe.alignment import SegmentMerger  # noqa

        return SegmentMerger().merge(transcript_segments, diar_segments)

    def _summarize(self, segments: list[MergedSegment]) -> str | None:
        from voxscribe.summarization import OllamaSummarizer  # noqa

        try:
            summarizer = OllamaSummarizer(
                model=self.cfg.ollama_model,
                host=self.cfg.ollama_host,
            )
            return summarizer.summarize(segments)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summarization failed (skipping): %s", exc)
            return None
