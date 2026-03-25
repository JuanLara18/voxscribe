"""Rich live display for real-time transcription."""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from voxscribe.realtime.streamer import StreamerState


MAX_LINES = 22          # Transcript lines visible at once
BORDER = "cyan"
STATUS_LISTENING  = "[green]● Listening[/]"
STATUS_PROCESSING = "[yellow]⚙ Processing[/]"
STATUS_ERROR      = "[red]✗ Error[/]"


class LiveDisplay:
    """Manages the Rich ``Live`` panel for real-time transcription.

    Intended to be used as a context manager::

        with LiveDisplay(model="small", device="cuda") as display:
            while running:
                display.update(streamer.get_state())

    Args:
        model: Model name shown in the title bar.
        device: Resolved device string shown in the title bar.
    """

    def __init__(self, model: str, device: str) -> None:
        self.model = model
        self.device = device
        self._start_time = time.monotonic()
        self._console = Console()
        self._live: Optional[Live] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> LiveDisplay:
        self._start_time = time.monotonic()
        self._live = Live(
            self._render(StreamerState()),
            console=self._console,
            refresh_per_second=8,
            screen=False,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)

    # ── Public update ─────────────────────────────────────────────────────────

    def update(self, state: StreamerState) -> None:
        """Redraw the panel with the latest streamer state."""
        if self._live:
            self._live.update(self._render(state))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, state: StreamerState) -> Panel:
        elapsed = time.monotonic() - self._start_time
        elapsed_str = _fmt_elapsed(elapsed)

        # ── Status indicator ──────────────────────────────────────────────────
        if state.error:
            status = f"{STATUS_ERROR}  {state.error[:70]}"
        elif state.is_processing:
            status = STATUS_PROCESSING
        else:
            status = STATUS_LISTENING

        # ── Title bar ─────────────────────────────────────────────────────────
        lang = (state.detected_language or "…").upper()
        device_label = self.device.upper()
        title = (
            f"[bold {BORDER}]VoxScribe Live[/]"
            f"  [dim]·[/]  [dim]{self.model}[/]"
            f"  [dim]·[/]  [{BORDER}]{lang}[/]"
            f"  [dim]·[/]  [dim]{device_label}[/]"
        )

        # ── Sub-title (status bar) ────────────────────────────────────────────
        n = len(state.segments)
        subtitle = f"[dim]{status}  ·  {elapsed_str}  ·  {n} segment{'s' if n != 1 else ''}[/]"

        # ── Transcript body ───────────────────────────────────────────────────
        body = Text(overflow="fold")
        lines = state.segments[-MAX_LINES:]

        if not lines:
            body.append("\n  Waiting for speech…", style="dim italic")
        else:
            for i, seg in enumerate(lines):
                is_last = i == len(lines) - 1
                if is_last:
                    body.append("\n  ▶ ", style=f"bold {BORDER}")
                    body.append(seg.text, style="bold white")
                else:
                    body.append("\n    ", style="")
                    body.append(seg.text, style="white")

        return Panel(
            body,
            title=title,
            subtitle=subtitle,
            border_style=BORDER,
            padding=(0, 1),
            expand=True,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_elapsed(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
