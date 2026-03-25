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


@app.command()
def live(
    # ── Transcription ─────────────────────────────────────────────────────
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help=(
                "Whisper model. [dim]'small' is a good default for real-time. "
                "'base' is faster, 'medium' more accurate.[/]"
            ),
            rich_help_panel="Transcription",
        ),
    ] = "small",
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
    translate: Annotated[
        bool,
        typer.Option(
            "--translate",
            help="Translate speech to English instead of transcribing in source language.",
            rich_help_panel="Transcription",
        ),
    ] = False,
    # ── Audio input ───────────────────────────────────────────────────────
    chunk: Annotated[
        float,
        typer.Option(
            "--chunk",
            help="Audio chunk duration in seconds. Lower = faster response, higher = more accurate.",
            rich_help_panel="Audio",
        ),
    ] = 4.0,
    input_device: Annotated[
        Optional[int],
        typer.Option(
            "--input-device", "-d",
            help="Microphone device index (see: voxscribe devices). Default: system default.",
            rich_help_panel="Audio",
        ),
    ] = None,
    # ── Global ────────────────────────────────────────────────────────────
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Live microphone transcription with on-screen subtitles.

    Captures audio from your microphone in real time and displays a rolling
    transcript on screen. Press [bold]Ctrl+C[/] to stop.

    [bold]Examples:[/bold]

      [green]# Start with auto-detect language (GPU auto-selected)[/green]
      voxscribe live

      [green]# Force Spanish, use medium model[/green]
      voxscribe live --lang es --model medium

      [green]# Translate any language to English[/green]
      voxscribe live --translate

      [green]# Slower chunks = better accuracy (less frequent updates)[/green]
      voxscribe live --chunk 6

      [green]# List available microphones first[/green]
      voxscribe devices
    """
    import threading

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Check sounddevice is available.
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        console.print(
            "[red]sounddevice is required for live mode.[/]\n"
            "Install it with: [bold]pip install voxscribe\\[realtime][/]"
        )
        raise typer.Exit(code=1)

    from voxscribe.realtime.capture import AudioCapture
    from voxscribe.realtime.display import LiveDisplay
    from voxscribe.realtime.streamer import LiveStreamer

    streamer = LiveStreamer(model=model, lang=lang, device=device, translate=translate)

    console.print(f"[cyan]Loading model '{model}'…[/]")
    try:
        resolved_device, compute_type = streamer.load_model()
    except Exception as exc:
        console.print(f"[red]Failed to load model:[/] {exc}")
        raise typer.Exit(code=1) from exc

    capture = AudioCapture(chunk_seconds=chunk, device=input_device)

    running = True

    def transcription_loop() -> None:
        while running:
            audio = capture.get_chunk(timeout=0.2)
            if audio is not None:
                streamer.process_chunk(audio)

    capture.start()
    worker = threading.Thread(target=transcription_loop, daemon=True)
    worker.start()

    console.print(
        f"[green]● Recording[/]  model=[bold]{model}[/]  "
        f"device=[bold]{resolved_device}[/] ({compute_type})  "
        f"chunk=[bold]{chunk}s[/]  "
        f"[dim]Ctrl+C to stop[/]"
    )

    try:
        with LiveDisplay(model=model, device=resolved_device) as display:
            while True:
                state = streamer.get_state()
                display.update(state)
                import time
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        capture.stop()
        worker.join(timeout=2)
        console.print("\n[cyan]Stopped.[/]")


@app.command()
def devices() -> None:
    """List available microphone / audio input devices.

    Use the index shown here with [bold]voxscribe live --input-device INDEX[/].
    """
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        console.print(
            "[red]sounddevice is required.[/] "
            "Install with: [bold]pip install voxscribe\\[realtime][/]"
        )
        raise typer.Exit(code=1)

    from rich.table import Table

    from voxscribe.realtime.capture import list_input_devices

    devs = list_input_devices()
    if not devs:
        console.print("[yellow]No input devices found.[/]")
        return

    table = Table(title="Available Microphones", border_style="cyan", show_lines=True)
    table.add_column("Index", style="bold cyan", justify="right")
    table.add_column("Name")
    table.add_column("Channels", justify="right")
    table.add_column("Default Sample Rate", justify="right")

    for d in devs:
        table.add_row(
            str(d["index"]),
            d["name"],
            str(d["channels"]),
            f"{d['default_samplerate']:,} Hz",
        )

    console.print(table)
    console.print("[dim]Use: voxscribe live --input-device INDEX[/]")


if __name__ == "__main__":
    app()
