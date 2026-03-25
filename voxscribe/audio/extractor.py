"""Audio extraction from video or audio files via FFmpeg.

Produces a 16 kHz mono PCM WAV file suitable for both ASR (Whisper) and
speaker diarization (pyannote / MFCC clustering).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Audio formats that can be passed directly without the -vn flag.
_AUDIO_ONLY_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}


class AudioExtractor:
    """Converts any audio or video file to a normalised WAV for downstream processing.

    Uses FFmpeg directly via subprocess for maximum compatibility. Supports
    any format that FFmpeg can decode — including all common video containers
    (mp4, mkv, mov, webm) and standalone audio files (mp3, flac, wav, …).

    Example::

        extractor = AudioExtractor()
        wav_path = extractor.extract("lecture.mp4", "output/audio.wav")
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        """
        Args:
            ffmpeg_path: Path or name of the FFmpeg executable.
        """
        self.ffmpeg_path = ffmpeg_path

    # ── Public API ────────────────────────────────────────────────────────

    def extract(
        self,
        input_path: str | Path,
        target_wav: str | Path,
        sample_rate: int = 16000,
        mono: bool = True,
    ) -> Path:
        """Extract and normalise audio from *input_path* into *target_wav*.

        Args:
            input_path: Source audio or video file.
            target_wav: Destination WAV path (created if missing).
            sample_rate: Target sample rate in Hz. Whisper requires 16 000 Hz.
            mono: Downmix to a single channel (required for diarization).

        Returns:
            Resolved :class:`~pathlib.Path` of the written WAV file.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            RuntimeError: If FFmpeg is unavailable or the conversion fails.
        """
        input_path = Path(input_path)
        target_wav = Path(target_wav)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not self._ffmpeg_available():
            raise RuntimeError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Install it with: brew install ffmpeg  /  apt install ffmpeg  /  "
                "https://ffmpeg.org/download.html"
            )

        target_wav.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(input_path, target_wav, sample_rate, mono)
        logger.debug("Running: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg conversion failed (exit {proc.returncode}):\n{proc.stderr}"
            )

        if not target_wav.exists():
            raise RuntimeError(f"FFmpeg exited 0 but did not create: {target_wav}")

        logger.info("Audio extracted → %s (%.1f MB)", target_wav, target_wav.stat().st_size / 1e6)
        return target_wav

    # ── Internals ─────────────────────────────────────────────────────────

    def _ffmpeg_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _build_command(
        self,
        input_path: Path,
        target_wav: Path,
        sample_rate: int,
        mono: bool,
    ) -> list[str]:
        is_audio_only = input_path.suffix.lower() in _AUDIO_ONLY_EXTENSIONS

        cmd = [self.ffmpeg_path, "-y", "-i", str(input_path)]

        if not is_audio_only:
            cmd.append("-vn")  # Strip video stream

        cmd += ["-ar", str(sample_rate)]

        if mono:
            cmd += ["-ac", "1"]

        cmd += ["-c:a", "pcm_s16le", str(target_wav)]
        return cmd
