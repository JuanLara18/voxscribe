# CLI Reference

## Synopsis

```
voxscribe [OPTIONS] INPUT
```

`INPUT` — path to any audio or video file: `mp4`, `mkv`, `mov`, `mp3`, `flac`, `wav`, `ogg`, …

---

## Options

### Transcription

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | `base` | Whisper model size. See [models](#models). |
| `--backend` | `faster-whisper` | `faster-whisper` or `whisperx` (word-level timestamps). |
| `--lang`, `-l` | auto | ISO-639-1 language code (`en`, `es`, `fr`, …). |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |
| `--compute-type` | `int8` | `int8` (fast CPU), `float16` (fast GPU), `float32` (precision). |

### Diarization

| Option | Default | Description |
|--------|---------|-------------|
| `--no-diarization` | — | Disable speaker diarization entirely. |
| `--hf-token` | `$HF_TOKEN` | HuggingFace token for pyannote. Also reads the `HF_TOKEN` env var. |
| `--min-speakers` | — | Lower bound on expected speaker count. |
| `--max-speakers` | — | Upper bound on expected speaker count. |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `output/` | Directory to write result files. |
| `--format`, `-f` | `md json` | Output format(s). Repeatable. |
| `--title` | input filename | Document title and output filename stem. |

Supported formats: `md`, `json`, `srt`, `vtt`, `txt`.

### Summarization

| Option | Default | Description |
|--------|---------|-------------|
| `--summarize` | — | Generate LLM summary via Ollama. |
| `--ollama-model` | `llama3.2` | Ollama model name. |
| `--ollama-host` | `http://localhost:11434` | Ollama API endpoint. |

### Global

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Enable debug logging. |
| `--version` | Print version and exit. |
| `--help` | Show help and exit. |

---

## Models

| Model | Speed | Languages | Notes |
|-------|-------|-----------|-------|
| `tiny` | Fastest | 99+ | Quick tests |
| `base` | Fast | 99+ | Default — good balance |
| `small` | Medium | 99+ | Better accuracy |
| `medium` | Slow | 99+ | High accuracy |
| `large-v3` | Slowest | 99+ | Maximum accuracy |
| `large-v3-turbo` | 5× faster than large-v3 | 99+ | **Recommended for production** |
| `distil-large-v3` | 6× faster than large-v3 | English only | Fastest English option |

---

## Environment variables

All options can be set via environment variables (prefix `VOXSCRIBE_`):

```bash
VOXSCRIBE_MODEL=large-v3-turbo
VOXSCRIBE_DEVICE=cuda
VOXSCRIBE_FORMATS=md,srt
HF_TOKEN=hf_...         # also accepted without prefix
```

Copy `.env.example` to `.env` to persist settings.

---

## Examples

```bash
# Basic (no extras needed)
voxscribe lecture.mp4

# SOTA quality — diarization + SRT subtitles
voxscribe interview.mp4 --model large-v3-turbo --hf-token $HF_TOKEN -f srt -f md

# Fast, no diarization, plain text
voxscribe lecture.wav --model tiny --no-diarization -f txt

# Multiple output formats
voxscribe podcast.mp3 --model small -f md -f json -f srt -f vtt

# Spanish audio, forced language
voxscribe reunión.mp4 --lang es --model medium

# Full SOTA: word timestamps + SOTA diarization + summary
voxscribe meeting.mp4 --backend whisperx --hf-token $HF_TOKEN --summarize

# Limit speaker count
voxscribe debate.mp4 --min-speakers 2 --max-speakers 4
```
