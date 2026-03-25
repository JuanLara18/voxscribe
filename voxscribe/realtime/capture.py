"""Real-time audio capture from microphone using sounddevice."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np


SAMPLE_RATE = 16_000  # Hz — Whisper's native sample rate


class AudioCapture:
    """Captures microphone audio in fixed-duration chunks.

    Runs a sounddevice InputStream in a background thread. Each fully
    accumulated chunk is placed in a thread-safe queue as a float32 numpy
    array of shape ``(chunk_samples,)`` at 16 kHz mono — exactly what
    faster-whisper expects.

    Args:
        chunk_seconds: Duration of each chunk in seconds.
        device: sounddevice device index or name. None = system default.
    """

    def __init__(
        self,
        chunk_seconds: float = 4.0,
        device: Optional[int | str] = None,
    ) -> None:
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(SAMPLE_RATE * chunk_seconds)
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Open and start the audio input stream."""
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=self.device,
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """Return the next ready audio chunk, or None if none available yet."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal callback (called from sounddevice audio thread) ─────────────

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
        with self._lock:
            self._buffer = np.concatenate([self._buffer, indata[:, 0]])
            # Emit complete chunks; keep any leftover samples for the next one.
            while len(self._buffer) >= self.chunk_samples:
                self._queue.put(self._buffer[: self.chunk_samples].copy())
                self._buffer = self._buffer[self.chunk_samples :]


# ── Device discovery ──────────────────────────────────────────────────────────


def list_input_devices() -> list[dict]:
    """Return metadata for all available input audio devices."""
    import sounddevice as sd

    return [
        {
            "index": i,
            "name": d["name"],
            "channels": d["max_input_channels"],
            "default_samplerate": int(d["default_samplerate"]),
        }
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
