"""faster-whisper transcription backend.

faster-whisper uses CTranslate2 as its runtime, which provides:
- ~4x speed improvement over openai/whisper on CPU
- ~5-8x on GPU with float16 / int8 quantization
- Lower memory footprint
- Full Python 3.10-3.13 compatibility (no pyproject.toml version conflicts)
- Built-in silero VAD to suppress hallucinations on silent segments

Model sizes supported (same weights as openai/whisper):
  tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo

``large-v3-turbo`` is the recommended default for production:
  5.4x faster than large-v3, <1% WER difference, all 99+ languages supported.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from voxscribe._utils import resolve_compute_type, resolve_device
from voxscribe.models import TranscriptSegment

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """Transcriber backed by the faster-whisper library.

    Lazy-loads the model on first call to :meth:`transcribe` so that
    instantiation is always cheap (no GPU memory allocated yet).

    Example::

        t = FasterWhisperTranscriber(model_size="large-v3-turbo", device="auto")
        segments, lang = t.transcribe(Path("audio.wav"))
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        """
        Args:
            model_size: Whisper model identifier (e.g. ``'base'``, ``'large-v3-turbo'``).
            device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
            compute_type: Quantization level.  ``'int8'`` (fast CPU),
                ``'float16'`` (fast CUDA), ``'float32'`` (precision).
                Use ``'auto'`` for device-appropriate selection.
            language: Force a specific language (ISO-639-1). ``None`` = auto-detect.
        """
        self.model_size = model_size
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.language = language
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(self, audio_path: Path) -> tuple[list[TranscriptSegment], str | None]:
        """Transcribe *audio_path* and return timestamped segments.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.

        Returns:
            Tuple of (segments, detected_language).

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            RuntimeError: On model load or inference failure.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()
        assert self._model is not None

        logger.info(
            "Transcribing with faster-whisper/%s on %s (%s) …",
            self.model_size,
            self.device,
            self.compute_type,
        )
        t0 = time.perf_counter()

        try:
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=5,
                vad_filter=True,       # silero VAD — reduces hallucinations
                vad_parameters={
                    "min_silence_duration_ms": 500,
                },
                word_timestamps=False,  # Not needed here; use WhisperX for word-level
            )
            # The generator must be consumed to run inference.
            segments = list(segments_gen)
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        logger.info(
            "Transcription done in %.1fs — %d segments, language='%s'",
            elapsed,
            len(segments),
            info.language,
        )

        result = [
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
            )
            for seg in segments
            if seg.text.strip()  # Drop empty segments
        ]
        return result, info.language

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            ) from exc

        logger.info(
            "Loading faster-whisper model '%s' on %s …", self.model_size, self.device
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Model loaded in %.1fs.", time.perf_counter() - t0)
