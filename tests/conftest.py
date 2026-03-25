"""Pytest fixtures shared across all test modules."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def tmp_audio_dir(tmp_path_factory) -> Path:
    """Session-scoped temp directory for generated audio fixtures."""
    return tmp_path_factory.mktemp("audio")


@pytest.fixture(scope="session")
def synthetic_wav(tmp_audio_dir) -> Path:
    """Generate a short (3 s) synthetic two-tone WAV file.

    The file contains a 440 Hz sine wave for the first 1.5 s and a
    880 Hz sine for the last 1.5 s — enough to exercise audio extraction
    and diarization code paths without real speech.
    """
    path = tmp_audio_dir / "test_audio.wav"
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    half = len(t) // 2
    signal = np.concatenate([
        0.5 * np.sin(2 * np.pi * 440 * t[:half]),
        0.5 * np.sin(2 * np.pi * 880 * t[half:]),
    ])
    samples = (signal * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())

    return path


@pytest.fixture
def sample_transcript_segments():
    """A minimal list of TranscriptSegment objects for unit tests."""
    from voxscribe.models import TranscriptSegment

    return [
        TranscriptSegment(start=0.0, end=2.5, text="Hello everyone."),
        TranscriptSegment(start=2.5, end=5.0, text="Welcome to the show."),
        TranscriptSegment(start=5.0, end=8.0, text="Today we discuss AI transcription."),
        TranscriptSegment(start=8.5, end=11.0, text="Let's get started."),
    ]


@pytest.fixture
def sample_diarization_segments():
    """A minimal list of DiarizationSegment objects for unit tests."""
    from voxscribe.models import DiarizationSegment

    return [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.5),
        DiarizationSegment(speaker="SPEAKER_01", start=5.5, end=11.0),
    ]


@pytest.fixture
def sample_merged_segments():
    """A minimal list of MergedSegment objects for exporter tests."""
    from voxscribe.models import MergedSegment

    return [
        MergedSegment(speaker="SPEAKER_00", start=0.0, end=2.5, text="Hello everyone."),
        MergedSegment(speaker="SPEAKER_00", start=2.5, end=5.0, text="Welcome to the show."),
        MergedSegment(speaker="SPEAKER_01", start=5.0, end=8.0, text="Today we discuss AI transcription."),
        MergedSegment(speaker="SPEAKER_01", start=8.5, end=11.0, text="Let's get started."),
    ]
