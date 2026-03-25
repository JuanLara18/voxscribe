"""VoxScribe command-line interface (Typer-based).

This module is a thin wrapper that parses CLI arguments into a
:class:`~voxscribe.config.VoxScribeConfig` and delegates to
:class:`~voxscribe.pipeline.Pipeline`.  All business logic lives in the
pipeline — the CLI only handles user interaction.

Entry point (registered in pyproject.toml)::

    voxscribe [OPTIONS] INPUT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from voxscribe import __version__

app = typer.Typer(
    name="voxscribe",
    help=(
        "VoxScribe — local, privacy-preserving transcription and speaker diarization.\n\n"
        "Transcribes any audio or video file using state-of-the-art local AI models.\n"
        "No data leaves your machine."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"VoxScribe {__version__}")
        raise typer.Exit()


@app.command()
def main(
    # ── Positional argument ───────────────────────────────────────────────
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to audio or video file (mp4, mkv, mov, mp3, flac, wav, ogg, …).",
            show_default=False,
        ),
    ],
    # ── Transcription options ─────────────────────────────────────────────
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help=(
                "Whisper model size. [dim]Choices: tiny, base, small, medium, "
                "large-v3, large-v3-turbo (recommended SOTA).[/]"
            ),
            rich_help_panel="Transcription",
        ),
    ] = "base",
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help=(
                "Transcription backend. [dim]'faster-whisper' (default, always available). "
                "'whisperx' for word-level timestamps (pip install voxscribe[alignment]).[/]"
            ),
            rich_help_panel="Transcription",
        ),
    ] = "faster-whisper",
    lang: Annotated[
        Optional[str],
        typer.Option(
            "--lang", "-l",
            help="Force language (ISO-639-1, e.g. 'en', 'es'). Default: auto-detect.",
            rich_help_panel="Transcription",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Compute device. [dim]'auto' selects CUDA if available.[/]",
            rich_help_panel="Transcription",
        ),
    ] = "auto",
    compute_type: Annotated[
        str,
        typer.Option(
            "--compute-type",
            help="Quantization. [dim]'int8' (fast CPU), 'float16' (fast GPU), 'float32' (precision).[/]",
            rich_help_panel="Transcription",
        ),
    ] = "int8",
    # ── Diarization options ───────────────────────────────────────────────
    no_diarization: Annotated[
        bool,
        typer.Option(
            "--no-diarization",
            help="Disable speaker diarization (faster, single-speaker output).",
            rich_help_panel="Diarization",
        ),
    ] = False,
    hf_token: Annotated[
        Optional[str],
        typer.Option(
            "--hf-token",
            envvar="HF_TOKEN",
            help=(
                "HuggingFace token for pyannote diarization (best accuracy). "
                "Also reads HF_TOKEN env var. "
                "[dim]Get yours at huggingface.co/settings/tokens[/]"
            ),
            rich_help_panel="Diarization",
        ),
    ] = None,
    min_speakers: Annotated[
        Optional[int],
        typer.Option(
            "--min-speakers",
            help="Minimum expected number of speakers.",
            rich_help_panel="Diarization",
        ),
    ] = None,
    max_speakers: Annotated[
        Optional[int],
        typer.Option(
            "--max-speakers",
            help="Maximum expected number of speakers.",
            rich_help_panel="Diarization",
        ),
    ] = None,
    # ── Output options ────────────────────────────────────────────────────
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output directory for result files.",
            rich_help_panel="Output",
        ),
    ] = Path("output"),
    format: Annotated[
        Optional[list[str]],
        typer.Option(
            "--format", "-f",
            help=(
                "Output format(s). Can be specified multiple times. "
                "[dim]Choices: md, json, srt, vtt, txt.[/]"
            ),
            rich_help_panel="Output",
        ),
    ] = None,
    title: Annotated[
        Optional[str],
        typer.Option(
            "--title",
            help="Document title (filename stem + Markdown heading). Defaults to input filename.",
            rich_help_panel="Output",
        ),
    ] = None,
    # ── Summarization options ─────────────────────────────────────────────
    summarize: Annotated[
        bool,
        typer.Option(
            "--summarize",
            help="Generate LLM summary via Ollama (pip install voxscribe[summarization]).",
            rich_help_panel="Summarization",
        ),
    ] = False,
    ollama_model: Annotated[
        str,
        typer.Option(
            "--ollama-model",
            help="Ollama model to use for summarization.",
            rich_help_panel="Summarization",
        ),
    ] = "llama3.2",
    ollama_host: Annotated[
        str,
        typer.Option(
            "--ollama-host",
            help="Ollama API host URL.",
            rich_help_panel="Summarization",
        ),
    ] = "http://localhost:11434",
    # ── Global options ────────────────────────────────────────────────────
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Transcribe and diarize an audio or video file locally.

    [bold]Examples:[/bold]

      [green]# Basic transcription[/green]
      voxscribe meeting.mp4

      [green]# High accuracy + subtitles[/green]
      voxscribe podcast.mp3 --model large-v3-turbo --hf-token $HF_TOKEN -f srt -f md

      [green]# Fast, no diarization, plain text[/green]
      voxscribe lecture.wav --model tiny --no-diarization -f txt

      [green]# Full SOTA pipeline with summary[/green]
      voxscribe interview.mp4 --model large-v3-turbo --hf-token $HF_TOKEN --summarize

      [green]# Python library[/green]
      from voxscribe import Transcriber
      result = Transcriber(model="large-v3-turbo", hf_token="...").run("audio.mp3")
      result.save("output/", formats=["srt", "md"])
    """
    # ── Logging setup ─────────────────────────────────────────────────────
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy third-party loggers unless in verbose mode.
    if not verbose:
        for noisy in ("faster_whisper", "whisperx", "pyannote", "numba", "torch"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Build config ──────────────────────────────────────────────────────
    from voxscribe.config import VoxScribeConfig  # noqa

    formats = format or ["md", "json"]

    try:
        cfg = VoxScribeConfig(
            model=model,
            backend=backend,
            language=lang,
            device=device,
            compute_type=compute_type,
            diarization=not no_diarization,
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            output_dir=str(output),
            formats=formats,
            title=title,
            summarize=summarize,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )
    except Exception as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    # ── Run pipeline ──────────────────────────────────────────────────────
    from voxscribe.pipeline import Pipeline  # noqa

    try:
        Pipeline(cfg).run(input_path)
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/] {exc}")
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(f"[red]Pipeline error:[/] {exc}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
