# Local voice assistant (LLM + STT + TTS + orb UI)

Fully **local** voice assistant for **macOS**: speak to the model, hear replies, and optionally use **macOS system controls** (apps, volume, screenshots, timers, Maps, and more). After models are cached, it runs **offline** — no API keys or cloud inference.

---

## What’s in the stack

| Piece | Technology |
|--------|----------------|
| **LLM** | [Llama 3.2 3B Instruct](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) GGUF **Q4_K_M** via **llama-cpp-python** (Apple **Metal** when `n_gpu_layers: -1`) |
| **STT** | **faster-whisper** (`small` by default, `int8`), language from `config.json` |
| **VAD** | **webrtcvad** for end-of-speech detection |
| **TTS** | **kokoro-onnx** (e.g. male voice `am_fenrir`) |
| **UI** | **Vite** + **TypeScript** + **Three.js** particle orb; real-time state over **WebSocket** |
| **Bridge** | `ws_server.py`: **HTTP 3000** serves `frontend/dist/`, **WS 8765** pushes `idle` / `listening` / `thinking` / `speaking` (+ mute API) |

The main entrypoint is **`chatbot_speech_to_speech.py`**, which loads models, records audio, streams LLM → TTS, and starts the HTTP + WebSocket servers.

---

## Repository layout

```
├── chatbot_speech_to_speech.py   # Full voice loop + pipeline + system commands
├── chatbot_text_to_speech.py     # Text in → speech out
├── chatbot_text_only.py          # Text chat only
├── ws_server.py                  # WebSocket + static UI server
├── start.command                 # macOS double-click launcher (Chrome + Python)
├── config.json                   # LLM / STT / TTS + system prompt
├── requirements_*.txt
├── frontend/
│   ├── src/                      # Orb UI source
│   ├── dist/                     # Production build (served offline; committed)
│   └── package.json
└── README.md
```

**Not committed** (too large or machine-local): `.venv311/`, `frontend/node_modules/`, **`kokoro-v1.0.onnx`**, **`voices-v1.0.bin`**, Hugging Face cache under `~/.cache/huggingface/`. Place ONNX/voices next to the project (or adjust paths in code/config) after downloading.

---

## Requirements

- **macOS** (launcher and several features use AppleScript / `open` / `screencapture`)
- **Python 3.11** (recommended; matches `start.command` and tested stack)
- **Node.js + npm** — only needed **once** to build the UI if `frontend/dist/` is missing or you change the frontend
- Microphone permission for Terminal / Python
- Optional: **Google Chrome** — `start.command` opens `http://localhost:3000` in Chrome if installed

---

## Installation

### 1. Python environment

```bash
cd /path/to/local-ai-voice-chatbot
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements_speech_to_speech.txt
```

For other variants:

```bash
pip install -r requirements_text_only.txt
pip install -r requirements_text_to_speech.txt
```

`llama-cpp-python` should be built with **Metal** on Apple Silicon for best performance (see upstream docs if you need a specific wheel/build).

### 2. Kokoro TTS weights (local files)

Download **`kokoro-v1.0.onnx`** and **`voices-v1.0.bin`** from the Kokoro ONNX distribution you use (e.g. project releases / Hugging Face) and put them in the **project root** (or update paths in `config.json` / loader code to match).

These files are **gitignored** and must be present locally.

### 3. LLM & Whisper caches

On first run with network, the app pulls the GGUF and Whisper assets via **Hugging Face Hub** and caches them under `~/.cache/huggingface/`. With cache populated, you can run **offline** (the loader is written to fall back to cache when the network is unavailable).

### 4. Frontend build (if needed)

If `frontend/dist/` is missing or you changed `frontend/src/`:

```bash
cd frontend
npm install
npm run build
```

The committed `frontend/dist/` is enough for **runtime** without Node.

---

## Running

### macOS (recommended)

Double-click **`start.command`** (or run it from Terminal). It:

- Checks `.venv311`
- Builds `frontend/dist` once if missing (requires `npm`)
- Starts `chatbot_speech_to_speech.py`
- Waits for port **3000**, then opens **Chrome** to the UI

Stop: close the Terminal window or press **Ctrl+C** (cleanup kills listeners on **3000** and **8765**).

### Manual

```bash
source .venv311/bin/activate
python chatbot_speech_to_speech.py
```

Open **http://localhost:3000** in a browser. The UI shows the orb and a **mute** control; status syncs over **ws://localhost:8765**.

---

## Configuration (`config.json`)

- **`llm`**: `repo_id`, GGUF `filename`, context, sampling, `prompt_behavior` (persona / style).
- **`tts`**: `model_file`, `voices_file`, `voice`, `speed`.
- **`stt`**: `model_size` (e.g. `small`), `language`.

Tweak `max_new_tokens`, `temperature`, or Whisper size to trade quality vs. speed.

---

## Features (speech build)

- Streaming LLM → chunked TTS → continuous playback (low gap between clauses).
- Orb states: **idle**, **listening**, **thinking**, **speaking** (plus optional demo effects from phrasing in the assistant logic).
- **Mute** via UI (HTTP API + WebSocket state).
- **macOS automation** (examples): open/quit apps, volume, screenshots, timers, Maps directions, Finder folders, date/time, memory/CPU/disk hints, clipboard-augmented prompts — implemented in Python with **AppleScript** / CLI tools.

---

## Ports

| Port | Service |
|------|---------|
| **3000** | Static UI + `/api/status`, `/api/mute` |
| **8765** | WebSocket state broadcast |

---

## Offline checklist

1. HF cache contains the **LLM GGUF** and **Whisper** model you configured.  
2. **`kokoro-v1.0.onnx`** and **`voices-v1.0.bin`** exist locally.  
3. **`frontend/dist/`** exists (ship with repo or run `npm run build` once).

---

## License

See [LICENSE](LICENSE).

---

## Contributing

Issues and PRs welcome. Keep **large weights** and **virtualenvs** out of Git; rebuild `frontend/dist` when changing the UI.
