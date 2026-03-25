"""Unit tests for voxscribe.config."""

from __future__ import annotations

import pytest

from voxscribe.config import VoxScribeConfig


class TestVoxScribeConfig:
    def test_defaults(self):
        cfg = VoxScribeConfig()
        assert cfg.model == "base"
        assert cfg.backend == "faster-whisper"
        assert cfg.diarization is True
        assert cfg.hf_token is None
        assert cfg.formats == ["md", "json"]
        assert cfg.summarize is False

    def test_custom_values(self):
        cfg = VoxScribeConfig(model="large-v3-turbo", device="cuda", formats=["srt"])
        assert cfg.model == "large-v3-turbo"
        assert cfg.device == "cuda"
        assert cfg.formats == ["srt"]

    def test_formats_validation(self):
        with pytest.raises(Exception):
            VoxScribeConfig(formats=["pdf"])

    def test_formats_comma_string(self):
        cfg = VoxScribeConfig.model_validate({"formats": "md,srt,json"})
        assert cfg.formats == ["md", "srt", "json"]

    def test_hf_token_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test123")
        cfg = VoxScribeConfig()
        assert cfg.hf_token == "hf_test123"

    def test_hf_token_prefixed_env(self, monkeypatch):
        monkeypatch.setenv("VOXSCRIBE_HF_TOKEN", "hf_prefixed")
        cfg = VoxScribeConfig()
        assert cfg.hf_token == "hf_prefixed"

    def test_unknown_model_does_not_raise(self):
        # Unknown models are warned about but not rejected (custom HF repos)
        cfg = VoxScribeConfig(model="custom/my-whisper-model")
        assert cfg.model == "custom/my-whisper-model"
