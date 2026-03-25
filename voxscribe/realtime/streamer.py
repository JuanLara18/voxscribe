"""Real-time transcription streamer — feeds audio chunks to faster-whisper."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from voxscribe._utils import resolve_compute_type, resolve_device

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class LiveSegment:
    """A single transcribed chunk of speech."""

    text: str
    language: str


@dataclass
class StreamerState:
    """Snapshot of streamer state — safe to read from any thread."""

    segments: list[LiveSegment] = field(default_factory=list)
    detected_language: Optional[str] = None
    is_processing: bool = False
    error: Optional[str] = None


# ── Streamer ──────────────────────────────────────────────────────────────────


class LiveStreamer:
    """Consumes audio chunks and produces transcribed text via faster-whisper.

    All public state access is thread-safe via an internal lock.  The display
    thread can call :meth:`get_state` at any time while the transcription
    thread calls :meth:`process_chunk`.

    Args:
        model: Whisper model size (``tiny`` / ``base`` / ``small`` / …).
        lang: Force ISO-639-1 language code, or None for auto-detection.
        device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
        translate: If True, translate to English instead of transcribing.
    """

    def __init__(
        self,
        model: str = "small",
        lang: Optional[str] = None,
        device: str = "auto",
        translate: bool = False,
    ) -> None:
        self.model_size = model
        self.lang = lang
        self.device_str = device
        self.translate = translate
        self._model = None
        self._state = StreamerState()
        self._lock = threading.Lock()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load_model(self) -> tuple[str, str]:
        """Load the Whisper model. Returns (device, compute_type) for display."""
        from faster_whisper import WhisperModel

        device = resolve_device(self.device_str)
        compute_type = resolve_compute_type("auto", device)
        logger.info(
            "Loading faster-whisper '%s' on %s (%s)…",
            self.model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
        )
        return device, compute_type

    # ── Transcription ─────────────────────────────────────────────────────────

    def process_chunk(self, audio: np.ndarray) -> None:
        """Transcribe one audio chunk and append non-empty results to state."""
        if self._model is None:
            return

        with self._lock:
            self._state.is_processing = True

        try:
            task = "translate" if self.translate else "transcribe"
            segments_iter, info = self._model.transcribe(
                audio,
                language=self.lang,
                task=task,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=1,   # greedy — fastest for real-time
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            # Materialize the lazy generator inside the try block.
            text = " ".join(seg.text.strip() for seg in segments_iter).strip()

            with self._lock:
                if not self._state.detected_language:
                    self._state.detected_language = info.language
                if text:
                    self._state.segments.append(
                        LiveSegment(text=text, language=info.language)
                    )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcription error: %s", exc)
            with self._lock:
                self._state.error = str(exc)
        finally:
            with self._lock:
                self._state.is_processing = False

    # ── State access ──────────────────────────────────────────────────────────

    def get_state(self) -> StreamerState:
        """Return a consistent snapshot of the current transcription state."""
        with self._lock:
            return StreamerState(
                segments=list(self._state.segments),
                detected_language=self._state.detected_language,
                is_processing=self._state.is_processing,
                error=self._state.error,
            )
