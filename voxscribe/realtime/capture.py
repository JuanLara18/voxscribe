"""Real-time audio capture from microphone using sounddevice."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np


SAMPLE_RATE = 16_000  # Hz — Whisper's native sample rate
_BLOCK_SIZE  = 1024   # samples per sounddevice callback (~64 ms at 16 kHz)


class AudioCapture:
    """Captures microphone audio and emits chunks sized by speech, not a fixed timer.

    Chunks are emitted in two cases:
    - **Silence flush**: the buffer has accumulated at least ``min_speech_seconds``
      of audio and the last ``silence_seconds`` have been silent → emit immediately.
      This makes transcription appear right when you finish a sentence.
    - **Max flush**: the buffer reaches ``max_chunk_seconds`` regardless of silence
      → guards against very long unbroken speech.

    Each emitted chunk is a float32 numpy array of shape ``(n_samples,)`` at 16 kHz
    mono — exactly what faster-whisper's ``transcribe()`` expects.

    Args:
        max_chunk_seconds:  Hard upper limit on chunk duration (seconds).
        min_speech_seconds: Minimum speech before a silence flush is considered.
        silence_seconds:    Trailing silence needed to trigger an early flush.
        silence_threshold:  RMS amplitude below which a block is considered silent.
        device:             sounddevice device index or name. None = system default.
    """

    def __init__(
        self,
        max_chunk_seconds: float = 6.0,
        min_speech_seconds: float = 1.0,
        silence_seconds: float = 0.6,
        silence_threshold: float = 0.015,
        device: Optional[int | str] = None,
    ) -> None:
        self.max_chunk_samples  = int(SAMPLE_RATE * max_chunk_seconds)
        self.min_speech_samples = int(SAMPLE_RATE * min_speech_seconds)
        self._silence_blocks    = int(SAMPLE_RATE * silence_seconds / _BLOCK_SIZE)
        self.silence_threshold  = silence_threshold
        self.device = device

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer = np.empty(0, dtype=np.float32)
        self._silent_block_count = 0
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
            blocksize=_BLOCK_SIZE,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        """Return the next ready audio chunk, or None if none is available yet."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal callback (audio thread) ─────────────────────────────────────

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # noqa: ANN001
        block = indata[:, 0]
        is_silent = float(np.sqrt(np.mean(block ** 2))) < self.silence_threshold

        with self._lock:
            self._buffer = np.concatenate([self._buffer, block])

            if is_silent:
                self._silent_block_count += 1
            else:
                self._silent_block_count = 0

            has_enough_speech = len(self._buffer) >= self.min_speech_samples
            silence_reached   = self._silent_block_count >= self._silence_blocks
            max_reached       = len(self._buffer) >= self.max_chunk_samples

            if (has_enough_speech and silence_reached) or max_reached:
                self._queue.put(self._buffer.copy())
                self._buffer = np.empty(0, dtype=np.float32)
                self._silent_block_count = 0


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
