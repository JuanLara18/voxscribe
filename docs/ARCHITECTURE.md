# Architecture

## Pipeline

VoxScribe processes audio and video through five sequential stages:

```mermaid
flowchart TD
    A["Input file\nmp4 · mkv · mov · mp3 · flac · wav …"] --> B["AudioExtractor\nFFmpeg → 16 kHz mono WAV"]
    B --> C["Transcriber\nfaster-whisper or WhisperX"]
    C --> D{Diarization?}
    D -- enabled --> E["Diarizer\npyannote 4.x or SimpleDiarizer"]
    E --> F["SegmentMerger\naligns text ↔ speakers"]
    D -- disabled --> F
    F --> G{Summarize?}
    G -- enabled --> H["OllamaSummarizer\nLlama 3.2 · Mistral · Qwen …"]
    H --> I["Exporters"]
    G -- disabled --> I
    I --> J["MD · JSON · SRT · VTT · TXT"]
```

## Data model

All components communicate through typed dataclasses defined in `voxscribe/models.py`. No raw dicts cross module boundaries.

```mermaid
classDiagram
    class WordTimestamp {
        +str word
        +float start
        +float end
        +float confidence
    }
    class TranscriptSegment {
        +float start
        +float end
        +str text
        +list~WordTimestamp~ words
    }
    class DiarizationSegment {
        +str speaker
        +float start
        +float end
    }
    class MergedSegment {
        +str speaker
        +float start
        +float end
        +str text
        +list~WordTimestamp~ words
    }
    class TranscriptResult {
        +list~MergedSegment~ segments
        +str language
        +float duration
        +str summary
        +save()
        +text()
        +speakers()
    }
    TranscriptSegment --> WordTimestamp
    MergedSegment --> WordTimestamp
    TranscriptResult --> MergedSegment
```

## Backend selection

The pipeline selects backends at runtime based on installed packages and configuration.

### Transcription

| `--backend` | Library | Notes |
|---|---|---|
| `faster-whisper` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Default. Always available. 4× faster than openai/whisper. |
| `whisperx` | [WhisperX](https://github.com/m-bain/whisperX) | Word-level timestamps via forced alignment. Optional. |

### Diarization

```mermaid
flowchart TD
    A{HF_TOKEN set?} -- No --> B[SimpleDiarizer\nMFCC + clustering\nno setup needed]
    A -- Yes --> C{pyannote-audio\ninstalled?}
    C -- No --> D[Warning + SimpleDiarizer\nfallback]
    C -- Yes --> E[PyannoteDiarizer\ncommunity-1 ~8% DER\nSOTA accuracy]
```

### WhisperX integrated path

When `--backend whisperx` + `--hf-token` are both set, the pipeline calls WhisperX's integrated diarization (transcription + forced alignment + pyannote speaker assignment in a single pass), bypassing the separate diarization and merger steps.

## Protocol pattern

Backends implement structural protocols — no inheritance required:

```python
class BaseTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> tuple[list[TranscriptSegment], str | None]: ...

class BaseDiarizer(Protocol):
    def diarize(self, audio_path: Path, ...) -> list[DiarizationSegment]: ...

class BaseExporter(Protocol):
    def export(self, segments: list[MergedSegment], output_path: Path, ...) -> None: ...
```

Adding a new backend = implement the protocol in a new file. No changes to `pipeline.py`.

## Configuration priority

```mermaid
flowchart LR
    A[Keyword args] --> Z[VoxScribeConfig]
    B[CLI flags] --> Z
    C[Env vars\nVOXSCRIBE_*] --> Z
    D[.env file] --> Z
    E[Defaults] --> Z
```

## Package structure

```
voxscribe/
  __init__.py            Public API: Transcriber, VoxScribeConfig, models
  cli.py                 Typer CLI (thin wrapper around Pipeline)
  pipeline.py            Orchestrator
  config.py              Pydantic Settings
  models.py              Shared dataclasses
  _utils.py              Internal helpers

  audio/
    extractor.py         AudioExtractor (FFmpeg)

  transcription/
    base.py              BaseTranscriber protocol
    faster_whisper.py    FasterWhisperTranscriber
    whisperx.py          WhisperXTranscriber

  diarization/
    base.py              BaseDiarizer protocol
    pyannote.py          PyannoteDiarizer
    simple.py            SimpleDiarizer (MFCC fallback)

  alignment/
    merger.py            SegmentMerger

  summarization/
    ollama.py            OllamaSummarizer

  exporters/
    base.py              BaseExporter protocol
    json_exporter.py
    markdown_exporter.py
    srt_exporter.py
    vtt_exporter.py
    txt_exporter.py
```
