"""Built-in MFCC-based speaker diarization (no external model required).

This diarizer works entirely offline with no HuggingFace token and no
additional model downloads. It uses classical signal-processing features
(MFCC + delta + delta²) and agglomerative clustering to segment audio into
speaker turns.

**Accuracy:** Competitive for clean, 2-4 speaker audio. Falls behind
pyannote/speaker-diarization-community-1 on crowded or noisy recordings.
Use :class:`~voxscribe.diarization.pyannote.PyannoteDiarizer` when accuracy
is critical and a HuggingFace token is available.

**Always available** — this is the zero-dependency fallback in the pipeline.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np

from voxscribe.models import DiarizationSegment

# Suppress librosa/numba noise that clutters normal user output.
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger(__name__)


class SimpleDiarizer:
    """Speaker diarizer using MFCC features + agglomerative clustering.

    Algorithm summary:
      1. Energy-based VAD to isolate speech regions.
      2. Extract 20 MFCC + Δ + ΔΔ features (60-dim) per segment.
      3. Estimate speaker count via the elbow method on Ward-linkage distortion.
      4. Cluster segments; post-process by merging adjacent same-speaker chunks.

    Example::

        d = SimpleDiarizer()
        segments = d.diarize(Path("audio.wav"), max_speakers=4)
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Args:
            device: Kept for API compatibility. MFCC extraction runs on CPU only.
        """
        # VAD parameters
        self.frame_length: float = 0.025    # 25 ms analysis frame
        self.frame_shift: float = 0.010     # 10 ms hop
        self.vad_threshold: float = 0.3     # normalised RMS threshold
        self.vad_pad_dur: float = 0.1       # padding around speech edges (s)
        self.min_segment_dur: float = 0.5   # discard segments shorter than this

        # Feature extraction
        self.n_mfcc: int = 20

        # Clustering
        self.merge_gap: float = 0.5         # merge same-speaker segments < this apart

    # ── Public API ────────────────────────────────────────────────────────

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Diarize *audio_path* into speaker turns.

        Args:
            audio_path: Path to a 16 kHz mono WAV file.
            min_speakers: Lower bound on speaker count (default: 1).
            max_speakers: Upper bound on speaker count (default: 10).

        Returns:
            List of :class:`~voxscribe.models.DiarizationSegment` sorted by start time.
        """
        import librosa  # noqa: PLC0415

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        min_speakers = max(1, min_speakers or 1)
        max_speakers = max(min_speakers, max_speakers or 10)

        logger.info("SimpleDiarizer: loading audio …")
        t0 = time.perf_counter()

        y, sr = librosa.load(str(audio_path), sr=None)
        logger.debug("Loaded %.1fs of audio at %d Hz.", len(y) / sr, sr)

        speech_segs = self._detect_speech(y, sr)
        if not speech_segs:
            logger.warning("No speech detected — returning empty diarization.")
            return []

        embeddings, meta = self._extract_features(y, sr, speech_segs)
        if len(meta) == 0:
            logger.warning("Feature extraction yielded no valid segments.")
            return []

        n_spk = self._estimate_speakers(embeddings, min_speakers, max_speakers)
        labels = self._cluster(embeddings, n_spk)
        segments = self._build_segments(meta, labels)
        segments = self._merge_adjacent(segments)

        elapsed = time.perf_counter() - t0
        logger.info(
            "SimpleDiarizer done in %.1fs — %d speakers, %d segments.",
            elapsed, n_spk, len(segments),
        )
        return segments

    # ── Signal processing ─────────────────────────────────────────────────

    def _detect_speech(self, y: np.ndarray, sr: int) -> list[dict]:
        """Energy-based VAD; returns list of {'start': float, 'end': float}."""
        import librosa  # noqa: PLC0415

        fl = int(self.frame_length * sr)
        hs = int(self.frame_shift * sr)

        energy = librosa.feature.rms(y=y, frame_length=fl, hop_length=hs)[0]
        e_range = np.max(energy) - np.min(energy)
        if e_range < 1e-10:
            return []
        energy_norm = (energy - np.min(energy)) / e_range
        is_speech = energy_norm > self.vad_threshold

        segments: list[dict] = []
        in_speech = False
        seg_start = 0.0
        duration = len(y) / sr

        for i, active in enumerate(is_speech):
            t = i * hs / sr
            if active and not in_speech:
                in_speech = True
                seg_start = max(0.0, t - self.vad_pad_dur)
            elif not active and in_speech:
                in_speech = False
                seg_end = min(duration, t + self.vad_pad_dur)
                if seg_end - seg_start >= self.min_segment_dur:
                    segments.append({"start": seg_start, "end": seg_end})

        if in_speech:
            seg_end = duration
            if seg_end - seg_start >= self.min_segment_dur:
                segments.append({"start": seg_start, "end": seg_end})

        return segments

    def _extract_features(
        self, y: np.ndarray, sr: int, speech_segs: list[dict]
    ) -> tuple[np.ndarray, list[dict]]:
        """Extract 60-dim MFCC+Δ+ΔΔ embeddings (one per speech segment)."""
        import librosa  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        embeddings, meta = [], []
        hs = int(self.frame_shift * sr)
        fl = int(self.frame_length * sr)

        for seg in speech_segs:
            s, e = int(seg["start"] * sr), int(seg["end"] * sr)
            chunk = y[s:e]
            if len(chunk) < sr * self.min_segment_dur:
                continue

            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=self.n_mfcc, hop_length=hs, n_fft=fl)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            feats = np.vstack([mfcc, delta, delta2]).T  # (frames, 60)

            if feats.shape[0] == 0:
                continue

            embeddings.append(np.mean(feats, axis=0))
            meta.append({"start": seg["start"], "end": seg["end"]})

        if not embeddings:
            return np.array([]), []

        X = np.vstack(embeddings)
        X = StandardScaler().fit_transform(X)
        return X, meta

    # ── Clustering ────────────────────────────────────────────────────────

    def _estimate_speakers(
        self, embeddings: np.ndarray, min_k: int, max_k: int
    ) -> int:
        """Elbow method on Ward-linkage distortion to choose speaker count."""
        from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

        n = len(embeddings)
        if n <= min_k:
            return min_k
        if min_k == max_k:
            return min_k

        max_k = min(max_k, n)
        distortions: list[float] = []
        ks = range(min_k, max_k + 1)

        for k in ks:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            model.fit(embeddings)
            d = sum(
                float(np.sum(np.linalg.norm(embeddings[model.labels_ == i] - embeddings[model.labels_ == i].mean(axis=0), axis=1) ** 2))
                for i in range(k)
                if np.sum(model.labels_ == i) > 0
            )
            distortions.append(d)

        if len(distortions) <= 1:
            return min_k

        diffs = np.diff(distortions)
        mx = np.max(np.abs(diffs))
        if mx > 0:
            diffs = diffs / mx

        for i, d in enumerate(diffs):
            if abs(d) < 0.2:
                return list(ks)[i + 1]

        return min(max(2, min_k), max_k)

    def _cluster(self, embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

        if len(embeddings) <= 1:
            return np.zeros(len(embeddings), dtype=int)

        model = AgglomerativeClustering(n_clusters=n_speakers, linkage="ward")
        return model.fit_predict(embeddings)

    # ── Post-processing ───────────────────────────────────────────────────

    def _build_segments(
        self, meta: list[dict], labels: np.ndarray
    ) -> list[DiarizationSegment]:
        segs = [
            DiarizationSegment(
                speaker=f"SPEAKER_{int(labels[i]):02d}",
                start=m["start"],
                end=m["end"],
            )
            for i, m in enumerate(meta)
            if i < len(labels)
        ]
        return sorted(segs, key=lambda s: s.start)

    def _merge_adjacent(self, segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
        """Merge consecutive same-speaker segments within *merge_gap* seconds."""
        if not segments:
            return []

        merged: list[DiarizationSegment] = []
        cur = DiarizationSegment(
            speaker=segments[0].speaker,
            start=segments[0].start,
            end=segments[0].end,
        )

        for seg in segments[1:]:
            if seg.speaker == cur.speaker and seg.start - cur.end < self.merge_gap:
                cur.end = seg.end
            else:
                merged.append(cur)
                cur = DiarizationSegment(speaker=seg.speaker, start=seg.start, end=seg.end)

        merged.append(cur)
        return merged
