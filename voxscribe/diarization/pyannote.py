"""pyannote.audio speaker diarization backend (SOTA accuracy).

pyannote/speaker-diarization-community-1 (pyannote.audio ≥ 3.x) achieves
~8% Diarization Error Rate on standard benchmarks — the best open-source
result for a general-purpose local model.

Requirements:
  1. ``pip install "voxscribe[diarization]"``
  2. A HuggingFace access token (https://huggingface.co/settings/tokens)
  3. Accept the model terms at:
     https://huggingface.co/pyannote/speaker-diarization-community-1

The token can be supplied via the ``hf_token`` constructor argument or
the ``HF_TOKEN`` environment variable (read automatically by VoxScribeConfig).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from voxscribe._utils import resolve_device
from voxscribe.models import DiarizationSegment

logger = logging.getLogger(__name__)

# The community-1 model from pyannote 4.x (released February 2026).
# Falls back to 3.1 for users on older pyannote installs.
_MODEL_COMMUNITY = "pyannote/speaker-diarization-community-1"
_MODEL_FALLBACK = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer:
    """Speaker diarizer backed by pyannote.audio.

    Lazy-loads the pipeline on first call to :meth:`diarize`.

    Example::

        d = PyannoteDiarizer(hf_token="hf_…")
        segments = d.diarize(Path("audio.wav"), max_speakers=3)
    """

    def __init__(self, hf_token: str, device: str = "auto") -> None:
        """
        Args:
            hf_token: HuggingFace access token (required).
            device: ``'auto'``, ``'cuda'``, or ``'cpu'``.
        """
        if not hf_token:
            raise ValueError(
                "hf_token is required for PyannoteDiarizer. "
                "Get yours at https://huggingface.co/settings/tokens"
            )
        self.hf_token = hf_token
        self.device = resolve_device(device)
        self._pipeline = None

    # ── Public API ────────────────────────────────────────────────────────

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Run pyannote diarization on *audio_path*.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.
            min_speakers: Optional lower bound on speaker count.
            max_speakers: Optional upper bound on speaker count.

        Returns:
            List of :class:`~voxscribe.models.DiarizationSegment` sorted by start time.

        Raises:
            ImportError: If pyannote.audio is not installed.
            RuntimeError: On pipeline failure.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_pipeline()
        assert self._pipeline is not None

        kwargs: dict = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        logger.info("Running pyannote diarization …")
        t0 = time.perf_counter()

        try:
            diarization = self._pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            raise RuntimeError(f"pyannote diarization failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        logger.info("pyannote diarization done in %.1fs.", elapsed)

        segments = [
            DiarizationSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            )
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s.start)

        n_speakers = len({s.speaker for s in segments})
        logger.info("Found %d speakers across %d segments.", n_speakers, len(segments))
        return segments

    # ── Internals ─────────────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is not installed. "
                "Run: pip install 'voxscribe[diarization]'"
            ) from exc

        logger.info("Loading pyannote pipeline (this may download ~1 GB on first run) …")
        t0 = time.perf_counter()

        # Try the latest community-1 model first; fall back to 3.1 for older installs.
        for model_id in (_MODEL_COMMUNITY, _MODEL_FALLBACK):
            try:
                pipeline = Pipeline.from_pretrained(
                    model_id,
                    use_auth_token=self.hf_token,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Model %s unavailable: %s", model_id, exc)
        else:
            raise RuntimeError(
                "Could not load any pyannote diarization model. "
                "Make sure you accepted the model terms at huggingface.co and "
                "that your token has read access."
            )

        if self.device == "cuda":
            import torch  # noqa: PLC0415

            pipeline.to(torch.device("cuda"))

        self._pipeline = pipeline
        logger.info("pyannote pipeline loaded in %.1fs.", time.perf_counter() - t0)
