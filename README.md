# VoxScribe

Local, privacy-preserving transcription and speaker diarization for any audio or video file.

Wraps state-of-the-art open-source models into a single CLI and Python library. No cloud. No data leaving your machine.

---

## Install

Requires Python 3.10+ and [FFmpeg](https://ffmpeg.org/download.html).

```bash
pip install -e .
```

**Optional extras** — install what you need:

```bash
pip install "voxscribe[diarization]"    # pyannote SOTA diarization (needs HF token)
pip install "voxscribe[alignment]"      # WhisperX word-level timestamps
pip install "voxscribe[summarization]"  # Ollama local LLM summarization
pip install "voxscribe[full]"           # everything
```

Verify: `python scripts/check_env.py`

---

## Usage

### CLI

```bash
# Basic
voxscribe lecture.mp4

# Production quality — subtitles + diarization
voxscribe interview.mp4 --model large-v3-turbo --hf-token $HF_TOKEN -f srt -f md

# Fast, no diarization
voxscribe lecture.wav --model tiny --no-diarization -f txt

# All formats + summary
voxscribe meeting.mp4 --model large-v3-turbo --hf-token $HF_TOKEN --summarize -f md -f srt -f json
```

Full CLI reference: [`docs/CLI.md`](docs/CLI.md)

### Python library

```python
from voxscribe import Transcriber

result = Transcriber(
    model="large-v3-turbo",
    hf_token="hf_...",
).run("interview.mp4")

result.save("output/", formats=["md", "srt"], title="interview")

print(result.language)   # "en"
print(result.speakers)   # ["SPEAKER_00", "SPEAKER_01"]
print(result.summary)    # LLM summary (if --summarize)
```

---

## What it uses

| Stage | Technology |
|---|---|
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 4× faster than openai/whisper |
| Word timestamps | [WhisperX](https://github.com/m-bain/whisperX) (optional) |
| Speaker diarization | [pyannote 4.x](https://github.com/pyannote/pyannote-audio) (~8% DER) or built-in MFCC fallback |
| Summarization | [Ollama](https://ollama.com) — Llama 3.2, Mistral, Qwen 3, … (optional) |
| Output | Markdown, JSON, SRT, WebVTT, plain text |

---

## Speaker diarization

VoxScribe picks the best available diarizer automatically:

| `HF_TOKEN` set | `pyannote-audio` installed | Result |
|---|---|---|
| No | — | Built-in MFCC diarizer (no setup needed) |
| Yes | No | Warning + MFCC fallback |
| Yes | Yes | **pyannote community-1 (~8% DER)** |

To get a token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → accept terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

---

## Documentation

- [Installation guide](docs/INSTALL.md) — FFmpeg, CUDA, HuggingFace token, Ollama
- [CLI reference](docs/CLI.md) — all options, models, environment variables
- [Architecture](docs/ARCHITECTURE.md) — pipeline, data model, backend selection

---

## Development

```bash
pip install -e ".[dev]"
pytest
python scripts/check_env.py
```

---

MIT License
