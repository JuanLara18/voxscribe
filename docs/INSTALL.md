# VoxScribe — Installation Guide

Detailed installation instructions for all platforms.

---

## Prerequisites

- **Python 3.10 – 3.13**
- **FFmpeg** (system package)
- 4 GB RAM minimum (8 GB+ recommended for large-v3 models)

### Install FFmpeg

| Platform | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH |

Verify: `ffmpeg -version`

---

## Install VoxScribe

### Option A — pip (standard)

```bash
git clone https://github.com/JuanLara18/voxscribe.git
cd voxscribe

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -e .
```

### Option B — uv (recommended, much faster)

```bash
pip install uv
git clone https://github.com/JuanLara18/voxscribe.git
cd voxscribe

uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

uv pip install -e .
```

### Install optional extras

```bash
# Best diarization (requires HuggingFace token)
pip install "voxscribe[diarization]"

# Word-level timestamps
pip install "voxscribe[alignment]"

# Local LLM summarization
pip install "voxscribe[summarization]"

# Everything
pip install "voxscribe[full]"
```

---

## GPU acceleration (optional)

VoxScribe automatically uses CUDA if a compatible GPU and PyTorch CUDA build are present.

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"`

---

## HuggingFace token (for pyannote diarization)

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Tokens](https://huggingface.co/settings/tokens) → **New token** (Read)
3. Accept the model terms at:
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
4. Set the token:
   ```bash
   # .env file (recommended)
   echo "HF_TOKEN=hf_your_token_here" >> .env

   # or environment variable
   export HF_TOKEN=hf_your_token_here
   ```

Without a token, VoxScribe uses the built-in MFCC diarizer which requires no setup.

---

## Ollama (for summarization)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the service
ollama serve

# Pull a model
ollama pull llama3.2     # fast, good quality (recommended)
ollama pull mistral       # strong instruction following
```

Then use `--summarize` in VoxScribe.

---

## Verify everything

```bash
python scripts/check_env.py
```

Expected output shows green checkmarks for all required components.

---

## Troubleshooting

### "FFmpeg not found"
- Verify installation: `ffmpeg -version`
- Windows: ensure FFmpeg's `bin/` folder is in your `PATH`

### "faster-whisper not installed"
- Run: `pip install faster-whisper`

### "pyannote model not found"
- Confirm you accepted the model terms on HuggingFace
- Check your token has **Read** permissions (not just Write)
- Try: `huggingface-cli login`

### "Ollama not reachable"
- Start the daemon: `ollama serve`
- Check it's running: `curl http://localhost:11434`

### Python 3.13 compatibility
VoxScribe uses faster-whisper (not openai-whisper), which has full Python 3.13
support. No special workarounds needed.
