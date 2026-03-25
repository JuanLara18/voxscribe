"""WhisperX transcription backend with forced alignment and integrated diarization.

WhisperX adds two capabilities on top of faster-whisper:
  1. **Forced alignment** via wav2vec2 — produces accurate word-level timestamps
     instead of Whisper's imprecise chunk-level ones.
  2. **Integrated speaker diarization** via pyannote — when a HuggingFace token
     is provided, ``transcribe_with_diarization`` assigns speakers at word level
     and returns :class:`~voxscribe.models.MergedSegment` objects directly,
     bypassing the separate diarization / merge steps in the pipeline.

Install::

    pip install "voxscribe[alignment]"
    # or:
    pip install whisperx

References:
    - https://github.com/m-bain/whisperX
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from voxscribe._utils import resolve_compute_type, resolve_device
from voxscribe.models import DiarizationSegment, MergedSegment, TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    """Transcriber + word-aligner backed by WhisperX.

    Two usage modes:

    * :meth:`transcribe` — transcription + word-level alignment only.
      Returns ``list[TranscriptSegment]`` with ``words`` populated.
      No HuggingFace token required.

    * :meth:`transcribe_with_diarization` — full integrated pipeline.
      Requires a HuggingFace token for pyannote.
      Returns ``list[MergedSegment]`` with speaker labels attached.

    Example::

        t = WhisperXTranscriber(model_size="large-v3-turbo", hf_token="hf_…")
        merged, lang = t.transcribe_with_diarization(Path("audio.wav"))
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self.language = language
        self.hf_token = hf_token
        self._model = None
        self._align_model = None
        self._align_metadata = None

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(self, audio_path: Path) -> tuple[list[TranscriptSegment], str | None]:
        """Transcribe and align *audio_path*; return word-level segments.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.

        Returns:
            Tuple of (segments with words, detected_language).
        """
        import whisperx  # noqa: PLC0415

        audio_path = Path(audio_path)
        audio = whisperx.load_audio(str(audio_path))

        self._load_model()
        t0 = time.perf_counter()
        result = self._model.transcribe(audio, batch_size=16, language=self.language)
        detected_lang = result.get("language", self.language)
        logger.info("WhisperX transcription done in %.1fs.", time.perf_counter() - t0)

        # Forced alignment for word-level timestamps.
        result = self._align(result, audio, detected_lang)

        return self._to_transcript_segments(result["segments"]), detected_lang

    def transcribe_with_diarization(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> tuple[list[MergedSegment], str | None]:
        """Full integrated pipeline: transcription + alignment + diarization.

        Requires a HuggingFace token (set via ``hf_token`` constructor arg or
        the ``HF_TOKEN`` environment variable).

        Args:
            audio_path: Path to a 16 kHz mono WAV file.
            min_speakers: Optional lower bound on speaker count.
            max_speakers: Optional upper bound on speaker count.

        Returns:
            Tuple of (merged segments with speaker labels, detected_language).

        Raises:
            ValueError: If no HuggingFace token is available.
        """
        import whisperx  # noqa: PLC0415

        if not self.hf_token:
            raise ValueError(
                "A HuggingFace token is required for WhisperX diarization. "
                "Pass hf_token= or set the HF_TOKEN environment variable."
            )

        audio_path = Path(audio_path)
        audio = whisperx.load_audio(str(audio_path))

        # Step 1: Transcribe.
        self._load_model()
        result = self._model.transcribe(audio, batch_size=16, language=self.language)
        detected_lang = result.get("language", self.language)

        # Step 2: Align.
        result = self._align(result, audio, detected_lang)

        # Step 3: Diarize and assign speakers at word level.
        logger.info("Running WhisperX diarization pipeline …")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token,
            device=self.device,
        )
        kwargs: dict = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize_model(audio, **kwargs)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._to_merged_segments(result["segments"]), detected_lang

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import whisperx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "whisperx is not installed. "
                "Run: pip install 'voxscribe[alignment]'"
            ) from exc
        logger.info("Loading WhisperX model '%s' …", self.model_size)
        self._model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )

    def _align(self, result: dict, audio, language: str | None) -> dict:
        """Run wav2vec2 forced alignment."""
        import whisperx  # noqa: PLC0415

        if language is None:
            logger.warning("Language unknown; skipping forced alignment.")
            return result

        if self._align_model is None or self._align_metadata is None:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
        return whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

    @staticmethod
    def _to_transcript_segments(raw: list[dict]) -> list[TranscriptSegment]:
        segments = []
        for seg in raw:
            words = None
            if "words" in seg:
                words = [
                    WordTimestamp(
                        word=w["word"],
                        start=w.get("start", seg["start"]),
                        end=w.get("end", seg["end"]),
                        confidence=w.get("score", 1.0),
                    )
                    for w in seg["words"]
                ]
            if seg.get("text", "").strip():
                segments.append(
                    TranscriptSegment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"].strip(),
                        words=words,
                    )
                )
        return segments

    @staticmethod
    def _to_merged_segments(raw: list[dict]) -> list[MergedSegment]:
        """Convert WhisperX output (with speaker labels) to MergedSegments."""
        segments = []
        for seg in raw:
            if not seg.get("text", "").strip():
                continue
            words = None
            if "words" in seg:
                words = [
                    WordTimestamp(
                        word=w["word"],
                        start=w.get("start", seg["start"]),
                        end=w.get("end", seg["end"]),
                        confidence=w.get("score", 1.0),
                    )
                    for w in seg["words"]
                ]
            speaker = seg.get("speaker", "SPEAKER_00")
            segments.append(
                MergedSegment(
                    speaker=speaker,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    words=words,
                )
            )
        return segments
