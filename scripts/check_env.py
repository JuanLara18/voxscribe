#!/usr/bin/env python3
"""VoxScribe environment check script.

Validates that all required and optional dependencies are available and
prints a colour-coded status table.  Run this after installation to confirm
your environment is set up correctly.

Usage::

    python scripts/check_env.py
    # or after pip install:
    python -c "import runpy; runpy.run_path('scripts/check_env.py')"
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from typing import NamedTuple

from rich.console import Console
from rich.table import Table

console = Console()


class Check(NamedTuple):
    name: str
    status: bool
    detail: str
    required: bool


def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 10)
    return Check(
        "Python ≥ 3.10",
        ok,
        f"Python {v.major}.{v.minor}.{v.micro}",
        required=True,
    )


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if not path:
        return Check("FFmpeg", False, "Not found in PATH", required=True)
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True
        )
        version = out.splitlines()[0].split("version")[1].split()[0] if "version" in out else "?"
        return Check("FFmpeg", True, f"v{version} at {path}", required=True)
    except Exception as exc:
        return Check("FFmpeg", False, str(exc), required=True)


def check_import(package: str, display_name: str, required: bool = False) -> Check:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "?")
        return Check(display_name, True, f"v{version}", required=required)
    except ImportError:
        install_hint = f"pip install {'voxscribe[full]' if not required else package}"
        return Check(display_name, False, f"Not installed — {install_hint}", required=required)


def check_cuda() -> Check:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return Check("CUDA GPU", True, f"{name} ({mem:.1f} GB)", required=False)
        return Check("CUDA GPU", False, "torch.cuda.is_available() = False", required=False)
    except ImportError:
        return Check("CUDA GPU", False, "torch not installed", required=False)


def check_hf_token() -> Check:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("VOXSCRIBE_HF_TOKEN")
    if token:
        masked = f"{token[:8]}…{token[-4:]}"
        return Check(
            "HF_TOKEN",
            True,
            f"{masked} (enables pyannote SOTA diarization)",
            required=False,
        )
    return Check(
        "HF_TOKEN",
        False,
        "Not set — falls back to built-in MFCC diarizer.  "
        "Get a token at huggingface.co/settings/tokens",
        required=False,
    )


def check_ollama() -> Check:
    try:
        import ollama  # noqa: F401
    except ImportError:
        return Check(
            "Ollama SDK",
            False,
            "Not installed — pip install 'voxscribe[summarization]'",
            required=False,
        )
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return Check("Ollama daemon", True, "Running at localhost:11434", required=False)
    except Exception:
        return Check(
            "Ollama daemon",
            False,
            "Not reachable at localhost:11434 — run 'ollama serve'",
            required=False,
        )


def main() -> None:
    console.rule("[bold blue]VoxScribe Environment Check[/]")

    checks: list[Check] = [
        # Required
        check_python(),
        check_ffmpeg(),
        check_import("faster_whisper", "faster-whisper", required=True),
        check_import("torch", "PyTorch", required=True),
        check_import("librosa", "librosa", required=True),
        check_import("sklearn", "scikit-learn", required=True),
        check_import("pydantic", "pydantic", required=True),
        check_import("typer", "typer", required=True),
        # Optional — quality improvements
        check_cuda(),
        check_hf_token(),
        check_import("pyannote.audio", "pyannote.audio (SOTA diarization)", required=False),
        check_import("whisperx", "whisperx (word timestamps)", required=False),
        check_ollama(),
    ]

    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Component", style="bold", min_width=30)
    table.add_column("Status", min_width=8)
    table.add_column("Detail")

    all_required_ok = True
    for c in checks:
        status_str = "[green]✓  OK[/]" if c.status else (
            "[red]✗  MISSING[/]" if c.required else "[yellow]–  OPTIONAL[/]"
        )
        if c.required and not c.status:
            all_required_ok = False
        table.add_row(c.name, status_str, c.detail)

    console.print(table)

    if all_required_ok:
        console.print(
            "\n[bold green]All required dependencies are satisfied.[/] "
            "Run [cyan]voxscribe --help[/] to get started."
        )
    else:
        console.print(
            "\n[bold red]Some required dependencies are missing.[/] "
            "See the details above and install them before using VoxScribe."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
