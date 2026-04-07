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
| **STT** | **faster-whisper** (`small` / `base`, `int8`) + **webrtcvad** end-of-speech |
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
| `stt.language` | `en`, `de`, … |
| `tts.voice` / `speed` | Kokoro voice ID and playback rate |

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
