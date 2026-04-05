# 🧠 Local Voice ChatBot with LLM + Whisper + TTS

A personal and fully-local chatbot with both **text** and **voice interaction**, designed to run entirely offline — no internet, no API keys, no cloud access required. 🛡️

Unlike typical chatbot solutions that rely on external servers or APIs, this project is built with **privacy**, **speed**, and **customizability** in mind.

---

## 🎯 Project Purpose

Over the past months, I've explored many open-source and commercial **Local AI Chatbot** projects, analyzing trade-offs between performance, accuracy, and system resources. The goal of this project was to:

- 🧠 Use the **most efficient and lightweight LLM** available for text generation  
- 🔊 Integrate a **natural-sounding, fast Text-to-Speech** (TTS) model  
- 🎙️ Add **speech input** support using small but powerful STT models  
- 💻 Run 100% locally — great for offline assistants, personal use, or edge devices  
- 🧩 Provide clean, modular Python code to expand or modify with ease

After testing many models and options, I selected the following:

- **LLM**: [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) — a highly optimized instruction-tuned model, fast and accurate even on consumer GPUs.
- **TTS**: [`Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) — a lightweight yet expressive voice model with speaker selection (British, American, male/female).
- **STT**: [`Whisper-tiny`](https://huggingface.co/openai/whisper-tiny) — small and surprisingly accurate for transcribing English speech.

---

## ✨ Cool Features

- ✅ **Prompt customization**: define assistant tone/behavior with a simple string
- ✅ **Voice selection**: switch between dozens of speakers for TTS (e.g., `af_heart`, `am_michael`, `bf_emma`, etc.)
- ✅ **Typing animation**: bot simulates thinking with a "typing..." animation
- ✅ **Streaming voice playback**: no `.wav` files saved, audio plays in realtime
- ✅ **VAD support**: automatically detects when the user has stopped speaking
- ✅ **Chat history**: short-term memory with logging to `chatlog.txt`
- ✅ **Modular config**: all models and settings live in `config.json` for easy tweaks
- ✅ **100% offline**: you can even disconnect Wi-Fi and everything will work 🚫🌐

This project is ideal for:
- 🧑‍💻 Developers who want to build or prototype voice assistants
- 🧘‍♂️ Privacy-conscious users who don’t want to send data to OpenAI/Gemini/etc
- 📚 Learners who want to understand local LLM + TTS + STT integration in Python

---

Ready to dive in? 🤖 Just pick the version that fits your needs and get chatting!

---

## 🧩 Versions

| Version | File | Description |
|---------|------|-------------|
| ✅ Basic | `chatbot_text_only.py` | Text-only input and output |
| ✅ Intermediate | `chatbot_text_to_speech.py` | Text input with spoken responses |
| ✅ Advanced | `chatbot_speech_to_speech.py` | Full voice interaction (Whisper + TTS) |

---

## 📦 Requirements

- Python 3.9+
- Anaconda (recommended)
- CUDA 11.8 for GPU acceleration (optional)
- ffmpeg for audio playback in TTS

---

## ⚙️ Installation

First, create a virtual environment:

```bash
conda create -n voicebot python=3.10
conda activate voicebot
```

Then install dependencies using the appropriate file:

```bash
pip install -r requirements_text_only.txt
pip install -r requirements_text_to_speech.txt
pip install -r requirements_speech_to_speech.txt
```

---

## 📁 Project Structure

```
project/
├── chatbot_text_only.py             # Text-only chatbot
├── chatbot_text_to_speech.py        # Text input with speech output
├── chatbot_speech_to_speech.py      # Full voice-based chatbot
├── config.json                      # Model configuration
├── requirements_*.txt               # Dependency files per version
└── chatlog.txt                      # Conversation logs
```

---

## 🧠 Used Models

- **LLM**: [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **TTS**: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (with voice selection)
- **STT**: [Whisper-Tiny](https://huggingface.co/openai/whisper-tiny)

---

## 🛠 Configuration

The `config.json` file holds all model settings, like model IDs, temperature, speaker voice, etc. Example:

```json
{
  "llm": {
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 150,
    "do_sample": true,
    "prompt_behavior": "You are a friendly and polite assistant who always replies clearly."
  },
  "tts": {
    "model_id": "hexgrad/Kokoro-82M",
    "speaker": "af_heart"
  },
  "stt": {
    "model_id": "tiny",
    "language": "en",
    "energy_threshold": 300,
    "pause_threshold": 0.8
  }
}
```

---

## ▶️ Running

### Text-only:

```bash
python chatbot_text_only.py
```

### With voice output:

```bash
python chatbot_text_to_speech.py
```

### Full voice interaction:

```bash
python chatbot_speech_to_speech.py
```

---

## 🔒 Fully Offline After First Setup

All models used in this project are **downloaded automatically on first run** and cached locally on your machine.  
After the initial download and setup, the chatbot runs **completely offline** without requiring any internet connection or API keys.  

> ⚠️ **Note:** The model files can be large (up to a few GB), so initial setup may take some time and bandwidth.

---

## 📓 Future Ideas

- 🔄 GUI interface
- 🌐 Web version using FastAPI
- 🌍 Multilingual support

---

## 📃 License

MIT License

---

## 🤝 Contribute

Feel free to contribute or expand the project. New voices or models are always welcome!