# Jarvis — Local AI Voice Assistant

> Fully **local**, fully **offline** voice assistant for **macOS**.
> Speak → LLM answers → you hear the reply. No API keys. No cloud.

---

## Download

**[⬇ Download Jarvis.dmg](https://github.com/Reezxy/Jarvis---Local-Voice-assistant/releases/latest)**

Mount the DMG, drag **Jarvis.app** into your project folder (next to `.venv311`), and double-click.
On first launch macOS will ask for **microphone access** — click Allow.

> **Requirements for the .app**
> - macOS 13 Ventura or later
> - The project folder with `.venv311`, models, and `chatbot_speech_to_speech.py` set up (see [Installation](#installation))

---

## What's in the stack

| Piece | Technology |
|---|---|
| **LLM** | [Llama 3.2 3B Instruct](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) Q4_K_M via **llama-cpp-python** (Apple **Metal** GPU) |
| **STT** | **faster-whisper** (configurable model + `beam_size`, `int8`) + **webrtcvad** end-of-speech |
| **TTS** | **kokoro-onnx** — voice `am_fenrir` (male EN) |
| **UI** | **Vite** + **TypeScript** + **Three.js** particle orb; real-time state over WebSocket |
| **Bridge** | `ws_server.py`: HTTP **:3000** serves `frontend/dist/`, WS **:8765** pushes state |
| **App** | Native **SwiftUI + AppKit** wrapper — launches backend, polls readiness, fullscreen WKWebView |

---

## Repository layout

```
├── chatbot_speech_to_speech.py   # Voice loop + LLM pipeline + system commands
├── ws_server.py                  # WebSocket + static HTTP server
├── config.json                   # LLM / STT / TTS + system prompt
├── requirements_speech_to_speech.txt
├── frontend/
│   ├── src/                      # Three.js orb source (TypeScript)
│   └── dist/                     # Pre-built UI (committed — no Node needed at runtime)
├── scripts/
│   └── benchmark_stt.py          # STT latency vs. accuracy sweep
└── JarvisApp/                    # Native macOS wrapper
    ├── JarvisApp.xcodeproj/
    ├── JarvisApp/                # Swift sources
    ├── notarize.sh               # Notarize + staple for distribution
    └── build_dmg.sh              # Package into a DMG (requires create-dmg)
```

**Not committed** (too large / machine-local): `.venv311/`, `kokoro-v1.0.onnx`, `voices-v1.0.bin`, HuggingFace cache under `~/.cache/huggingface/`.

---

## Installation

### 1 — Python environment

```bash
cd /path/to/local-ai-voice-chatbot
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements_speech_to_speech.txt
```

`llama-cpp-python` should be built with **Metal** on Apple Silicon (see upstream docs for the correct wheel / cmake flags).

### 2 — Kokoro TTS weights

Download **`kokoro-v1.0.onnx`** and **`voices-v1.0.bin`** and place them in the **project root**.
These are gitignored and must be present locally before first run.

### 3 — LLM + Whisper (auto-downloaded on first run)

On first run with a network connection the app pulls the GGUF and Whisper weights via **Hugging Face Hub** into `~/.cache/huggingface/`. After that it runs fully **offline**.

### 4 — Frontend (already built)

The committed `frontend/dist/` is sufficient. Rebuild only if you change `frontend/src/`:

```bash
cd frontend && npm install && npm run build
```

---

## Running

### Option A — Jarvis.app (recommended)

1. Build or download the app (see [Download](#download)).
2. Place `Jarvis.app` in the project root (next to `.venv311`).
3. Double-click. The app:
   - Requests **microphone permission** on first launch
   - Starts the Python backend automatically
   - Shows the orb UI once port 3000 is ready
   - Streams backend logs via **Jarvis → Show Logs** (⌘⇧L)
   - Restarts the backend via **Jarvis → Restart Backend** (⌘⇧R)

### Option B — Terminal (no app)

```bash
source .venv311/bin/activate
python chatbot_speech_to_speech.py
# Open http://localhost:3000 in your browser
```

---

## Configuration (`config.json`)

| Key | What it controls |
|---|---|
| `llm.repo_id` / `filename` | Which GGUF model to load |
| `llm.n_gpu_layers` | `-1` = full Metal offload |
| `llm.temperature` / `max_new_tokens` | Generation quality vs. speed |
| `stt.model_size` | `tiny` / `base` / `small` — accuracy vs. latency |
| `stt.model` | Optional override: any faster-whisper size **or** HF repo id (e.g. `distil-whisper/distil-small.en`). Takes precedence over `model_size` |
| `stt.beam_size` | Whisper decode width — `1` greedy (fastest) … `5` (old default, slowest) |
| `stt.compute_type` | `int8` (default), `int8_float16`, `float32` |
| `stt.language` | `en`, `de`, … |
| `tts.voice` / `speed` | Kokoro voice ID and playback rate |

---

## STT speed vs. accuracy

Transcription sits directly on the critical path: nothing else in a turn starts
until Whisper returns. Two knobs move it.

**`stt.beam_size`** — beam search explores *n* candidate decodings in parallel.
The old hard-coded `5` roughly triples decode work versus greedy for a gain that
is mostly invisible on short, command-shaped utterances. The default is now `2`;
drop to `1` if you want the floor.

**`stt.model` / `stt.model_size`** — smaller models are dramatically faster but
degrade on accents, proper nouns, and noisy rooms.

| Model | Relative speed | Where it breaks down |
|---|---|---|
| `tiny.en` | fastest | Proper nouns, app names, accented speech — expect to repeat yourself |
| `base.en` | fast | Good floor for short commands; struggles with long dictation |
| `distil-whisper/distil-small.en` | fast, `small`-class accuracy | English only; extra ~500 MB download |
| `small` (default) | slowest of these | Best accuracy here, noticeably more latency per turn |

The `.en`-suffixed and distil models are **English-only** — keep `small` (or
another multilingual size) if you set `stt.language` to anything but `en`.

Rule of thumb: pair a *smaller model* with a *larger beam* rather than the other
way round. `base.en` + `beam_size: 2` beats `small` + `beam_size: 1` on both axes
for most command traffic.

### Measure it on your machine

Numbers depend on your CPU, mic, and accent, so benchmark rather than trust a
table:

```bash
source .venv311/bin/activate

# Synthesise test clips with the project's own Kokoro voice (no recordings needed)
python scripts/benchmark_stt.py --synthesise

# Or use your own: samples/foo.wav (16-bit mono) + samples/foo.txt (reference text)
python scripts/benchmark_stt.py --samples samples/ --beams 1 2 5
```

It sweeps every (model, beam_size) pair and prints mean latency per clip,
real-time factor, and word error rate against the reference transcripts. Pick
the fastest row whose WER you can live with, then set it in `config.json`.

> Synthesised speech is clean and consistent — it is a good *relative* ranking
> of configurations, but it flatters every model's WER. For an absolute read on
> how a config handles your voice and your room, record real clips.

---

## Features

- **Streaming** LLM → chunked TTS → gapless playback
- **Orb states**: idle · listening · thinking · speaking (+ demo effects)
- **Mute** via orb UI
- **macOS automation** — open/quit apps, volume, screenshots, timers, Maps, Finder, clipboard augmentation — all via AppleScript / CLI, no LLM roundtrip
- **STT overlay** in the native app — shows the transcribed text as a small pill in the corner
- **Live logs** window in the native app

---

## Build the app from source

```bash
open JarvisApp/JarvisApp.xcodeproj
# Select "My Mac" target → ⌘R
```

**Distribute:**

```bash
# 1. Archive in Xcode: Product → Archive → Distribute App → Developer ID
# 2. Notarize:
cd JarvisApp
./notarize.sh /path/to/Jarvis.app your@apple.id TEAMID xxxx-xxxx-xxxx-xxxx
# 3. DMG:
./build_dmg.sh /path/to/Jarvis.app
```

---

## Ports

| Port | Service |
|---|---|
| **3000** | Static UI + `/api/status` `/api/mute` |
| **8765** | WebSocket state broadcast |

---

## Offline checklist

1. HF cache has the **GGUF** and **Whisper** model configured in `config.json`
2. `kokoro-v1.0.onnx` and `voices-v1.0.bin` are in the project root
3. `frontend/dist/` exists

---

## License

See [LICENSE](LICENSE).

---

## Website: https://jarvis-mac.lovable.app

---

## Contributing

Issues and PRs welcome. Keep large weights and virtualenvs out of Git.
