"""
Local Voice Assistant — Jarvis Edition
─────────────────────────────────────────────────────────────────────────────
LLM  : Llama-3.2-3B-Instruct Q4_K_M via llama-cpp-python (Apple Metal GPU)
TTS  : Kokoro-82M ONNX  ·  voice: am_fenrir (male)  ·  ~200 ms/sentence
STT  : faster-whisper (configurable model + beam_size) + int8 + VAD filter
Wake : openWakeWord 'hey_jarvis' — always-on, near-zero CPU while idle
─────────────────────────────────────────────────────────────────────────────
Pipeline   : LLM-stream → TTS-stream → SeamlessPlayer (zero-gap audio)
System cmds: volume, apps, screenshot, timer — executed locally, no LLM
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from neural_memory import MemorySystem
from neural_memory.providers import LocalProvider

import ws_server

# ── Data directory ────────────────────────────────────────────────────────────
# When running inside the macOS app, the Swift wrapper sets JARVIS_DATA_DIR to
# ~/Library/Application Support/Jarvis/ so models and config survive app updates.
# When running locally (start.command / CLI), falls back to the script directory.
_DATA_DIR = Path(os.environ["JARVIS_DATA_DIR"]) if "JARVIS_DATA_DIR" in os.environ else Path(__file__).parent

# ── Constants ─────────────────────────────────────────────────────────────────
# config.json: user's copy in data dir first, bundled default as fallback
CONFIG_PATH  = _DATA_DIR / "config.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = Path(__file__).parent / "config.json"
MEMORY_DIR   = str(_DATA_DIR / "jarvis_memory_db")

WAKE_TIMEOUT = 120   # seconds of silence → enter wake-word mode

# openWakeWord runs on 80 ms frames of 16 kHz int16 audio.
WAKE_CHUNK_SAMPLES   = 1_280
WAKE_MODEL_DEFAULT   = "hey_jarvis"
WAKE_THRESHOLD_DEFAULT = 0.5
SAMPLE_RATE  = 16_000
TTS_RATE     = 24_000
FRAME_MS     = 30
FRAME_SIZE   = int(SAMPLE_RATE * FRAME_MS / 1_000)

# Adaptive silence: short for quick commands, longer once you've been speaking a while
SILENCE_CUTOFF_SHORT_MS  = 520
SILENCE_CUTOFF_LONG_MS   = 950
LONG_SPEECH_THRESHOLD_MS = 2_500   # use long cutoff after 2.5 s of speech

PLAYER_BLOCKSIZE = 4_096

# Whisper decoding: greedy-ish by default. See README "STT speed vs accuracy".
STT_BEAM_SIZE_DEFAULT = 2

# Barge-in: how much *continuous* speech is needed to cut Jarvis off mid-sentence.
# Long enough that a cough or a door doesn't stop him, short enough to feel instant.
BARGE_IN_SUSTAIN_MS = 360
# Ignore the mic for a moment after playback starts — the output device ramping
# up otherwise reads as a burst of "speech" on the very first frames.
BARGE_IN_GRACE_MS   = 350
# int16 RMS gate on top of the VAD. TTS bleeding back through the mic is much
# quieter than someone actually talking to the machine; this is what separates them.
BARGE_IN_RMS_FLOOR  = 900.0
BARGE_IN_VAD_LEVEL  = 3

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_RE   = re.compile(r"(?<=[,;:])\s+")
MIN_CLAUSE_WORDS = 8


# ── Helpers ───────────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} …", flush=True)

    def _hook(count: int, block: int, total: int) -> None:
        pct = min(100, count * block * 100 // max(total, 1))
        sys.stdout.write(f"\r  {pct:3d}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def _clean(text: str) -> str:
    """Strip LLM artefacts: <think> tags, markdown symbols, excess newlines."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Seamless audio player ──────────────────────────────────────────────────────
class SeamlessPlayer:
    """
    Plays a continuous stream of float32 mono audio fed from a queue.
    Uses sounddevice.OutputStream with a callback so chunks are joined
    at sample level — no gap, click, or silence between sentences.
    """

    def __init__(self, sample_rate: int = TTS_RATE) -> None:
        self._sr      = sample_rate
        self._buf     = np.empty(0, dtype=np.float32)
        self._lock    = threading.Lock()
        self._done    = threading.Event()
        self._feeding = True
        self._stream: Optional[sd.OutputStream] = None

    def start(self) -> None:
        self._done.clear()
        self._feeding = True
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=PLAYER_BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    def feed(self, audio: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate((self._buf, audio.ravel()))

    def mark_done(self) -> None:
        self._feeding = False

    def wait(self) -> None:
        self._done.wait()
        self._close()

    def stop(self) -> None:
        self._feeding = False
        self._done.set()
        self._close()

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int, _time, _status) -> None:
        with self._lock:
            have = len(self._buf)
            if have >= frames:
                outdata[:, 0] = self._buf[:frames]
                self._buf = self._buf[frames:]
            elif have > 0:
                outdata[:have, 0] = self._buf
                outdata[have:, 0] = 0.0
                self._buf = np.empty(0, dtype=np.float32)
                if not self._feeding:
                    threading.Timer(0.05, self._done.set).start()
            else:
                outdata[:, 0] = 0.0
                if not self._feeding:
                    self._done.set()


# ── Wake word ──────────────────────────────────────────────────────────────────
class WakeWordDetector:
    """
    Continuous "Hey Jarvis" detection via openWakeWord.

    Replaces transcribing every idle clip with full Whisper: the model is a few
    hundred KB running on 80 ms frames, so idle listening costs a fraction of a
    core instead of a Whisper decode every couple of seconds — the CPU/GPU
    headroom belongs to actual turns.
    """

    def __init__(self, model, names: list[str], threshold: float) -> None:
        self._model     = model
        self._names     = names
        self._threshold = threshold

    @classmethod
    def create(cls, cfg: dict) -> "Optional[WakeWordDetector]":
        """
        Build a detector from the `wake_word` config block.

        Returns None (with a printed reason) when openWakeWord is unavailable or
        its models can't be fetched — the assistant then simply stays in
        always-on listening instead of failing to start.
        """
        if not cfg.get("enabled", True):
            return None

        names     = cfg.get("models") or [WAKE_MODEL_DEFAULT]
        threshold = float(cfg.get("threshold", WAKE_THRESHOLD_DEFAULT))
        framework = cfg.get("inference_framework", "onnx")

        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            print("[Wake] openwakeword not installed — wake-word mode disabled.")
            print("       pip install -r requirements_speech_to_speech.txt")
            return None

        print(f"[Wake] Loading openWakeWord {names} …")
        extra = {}
        if "vad_threshold" in cfg:
            extra["vad_threshold"] = float(cfg["vad_threshold"])
        try:
            model = Model(wakeword_models=names, inference_framework=framework, **extra)
        except Exception:
            # First run: pretrained weights aren't in the openWakeWord cache yet.
            print("[Wake] Models not cached — downloading (requires internet) …")
            try:
                openwakeword.utils.download_models()
                model = Model(wakeword_models=names, inference_framework=framework, **extra)
            except Exception as exc:                       # noqa: BLE001
                print(f"[Wake] Could not load wake-word models ({exc}) — "
                      "wake-word mode disabled.")
                return None

        loaded = list(model.prediction_buffer.keys()) or names
        print(f"[Wake] Ready — {loaded} @ threshold {threshold}")
        return cls(model, loaded, threshold)

    def listen(self, should_abort=None) -> bool:
        """
        Block until the wake word is detected. Returns False if `should_abort()`
        goes true first (mute) or the mic can't be opened.
        """
        audio_q: queue.Queue[bytes] = queue.Queue()

        def _cb(indata: np.ndarray, *_) -> None:
            audio_q.put(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=WAKE_CHUNK_SAMPLES,
                dtype="int16", channels=1, callback=_cb,
            )
        except Exception as exc:                           # noqa: BLE001
            print(f"[Wake] Mic unavailable ({exc})")
            return False

        self._model.reset()
        with stream:
            while True:
                if should_abort is not None and should_abort():
                    return False
                try:
                    chunk = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                samples = np.frombuffer(chunk, dtype="int16")
                scores  = self._model.predict(samples)
                if any(scores.get(n, 0.0) >= self._threshold for n in self._names):
                    self._model.reset()   # don't re-fire on the same utterance
                    return True


# ── Barge-in listener ──────────────────────────────────────────────────────────
class BargeInListener:
    """
    Mic watchdog that runs *while Jarvis is speaking* so the user can cut him off.

    Deliberately cheap: webrtcvad on 30 ms frames plus an RMS gate — no Whisper,
    no LLM. It fires only after BARGE_IN_SUSTAIN_MS of *continuous* qualifying
    speech, which is what keeps background noise and Jarvis's own voice bleeding
    into the mic from triggering a false interrupt.

    The frames that caused the trigger are kept in `prefix` so the first word of
    the interrupting utterance isn't lost when recording takes over.
    """

    def __init__(self, on_trigger, *, sustain_ms: int = BARGE_IN_SUSTAIN_MS,
                 grace_ms: int = BARGE_IN_GRACE_MS,
                 rms_floor: float = BARGE_IN_RMS_FLOOR,
                 vad_level: int = BARGE_IN_VAD_LEVEL) -> None:
        self._on_trigger = on_trigger
        self._sustain_ms = sustain_ms
        self._grace_ms   = grace_ms
        self._rms_floor  = rms_floor
        self._vad        = webrtcvad.Vad(vad_level)
        self._q: queue.Queue[bytes] = queue.Queue()
        self._stop       = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.triggered   = threading.Event()
        self.prefix: bytes = b""

    def start(self) -> None:
        self._stop.clear()
        self.triggered.clear()
        self.prefix = b""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _qualifies(self, frame: bytes) -> bool:
        if not self._vad.is_speech(frame, SAMPLE_RATE):
            return False
        samples = np.frombuffer(frame, dtype="int16").astype(np.float32)
        return float(np.sqrt(np.mean(samples * samples))) >= self._rms_floor

    def _run(self) -> None:
        def _cb(indata: np.ndarray, *_) -> None:
            self._q.put(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE,
                dtype="int16", channels=1, callback=_cb,
            )
        except Exception:
            return   # mic busy or unavailable — barge-in is simply off this turn

        frames: list[bytes] = []
        speech_ms   = 0
        grace_until = time.time() + self._grace_ms / 1_000.0
        try:
            with stream:
                while not self._stop.is_set():
                    try:
                        frame = self._q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if time.time() < grace_until:
                        continue
                    if self._qualifies(frame):
                        frames.append(frame)
                        speech_ms += FRAME_MS
                        if speech_ms >= self._sustain_ms:
                            self.prefix = b"".join(frames)
                            self.triggered.set()
                            try:
                                self._on_trigger()
                            except Exception:
                                pass
                            return
                    else:
                        frames.clear()
                        speech_ms = 0
        except Exception:
            pass


# ── Voice Assistant ────────────────────────────────────────────────────────────
class VoiceAssistant:
    def __init__(self) -> None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            self.cfg: dict = json.load(f)

        self._load_llm()
        self._load_tts()
        self._load_stt()

        wake_cfg = self.cfg.get("wake_word", {})
        self._wake_timeout: float = float(wake_cfg.get("timeout_s", WAKE_TIMEOUT))
        self._wake = WakeWordDetector.create(wake_cfg)

        self.vad = webrtcvad.Vad(3)
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self.history: list[dict] = []
        self.system_prompt: str = self.cfg["llm"].get(
            "prompt_behavior",
            "You are Jarvis, a helpful and concise voice assistant. "
            "Your name is Jarvis. The user's name is Felix. "
            "Address the user naturally as 'Sir' or 'Felix' when it fits. "
            "If asked for your name, say your name is Jarvis. "
            "Keep answers brief and conversational. No bullet points or markdown.",
        )
        self._stop_speak = threading.Event()

        # Serialises every path that opens an output stream. The proactive
        # monitor runs on its own thread, so without this a battery or calendar
        # announcement can land on the output device while the main loop is
        # already speaking a reply — two streams, one speaker. Reentrant so a
        # nested speak from the same thread can't deadlock.
        self._speak_lock = threading.RLock()

        # Barge-in: set while an interrupt is pending, with the audio that
        # triggered it so record_audio() can keep the user's first word.
        bi = self.cfg.get("barge_in", {})
        self._barge_in_enabled: bool = bool(bi.get("enabled", True))
        self._barge_in_opts = {
            "sustain_ms": int(bi.get("sustain_ms", BARGE_IN_SUSTAIN_MS)),
            "grace_ms":   int(bi.get("grace_ms", BARGE_IN_GRACE_MS)),
            "rms_floor":  float(bi.get("rms_floor", BARGE_IN_RMS_FLOOR)),
            "vad_level":  int(bi.get("vad_level", BARGE_IN_VAD_LEVEL)),
        }
        self._barge_in = threading.Event()
        self._barge_in_audio: bytes = b""

        # Persistent neural memory
        from neural_memory import MemoryConfig
        self._mem = MemorySystem(
            provider=LocalProvider(),
            config=MemoryConfig(persist_directory=MEMORY_DIR),
        )

        # Proactive background monitor
        threading.Thread(target=self._proactive_loop, daemon=True).start()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_llm(self) -> None:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from llama_cpp import Llama

        c = self.cfg["llm"]
        repo_id  = c["repo_id"]
        filename = c["filename"]
        print(f"[LLM] Loading {repo_id}  ({filename}) …")

        cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if cached and Path(cached).is_file():
            model_path = cached
            print("[LLM] Found in local cache — skipping network.")
        else:
            # try_to_load_from_cache missed it — try local_files_only before going to the network
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, local_files_only=True
                )
                print("[LLM] Found in HF cache (local_files_only) — skipping network.")
            except Exception:
                print("[LLM] Not cached — downloading from HuggingFace …")
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=c.get("n_gpu_layers", -1),
            n_ctx=c.get("n_ctx", 4096),
            verbose=False,
        )
        self._llm_cfg = c
        print("[LLM] Ready  (Metal GPU layers active)")

    def _load_tts(self) -> None:
        from kokoro_onnx import Kokoro

        c = self.cfg["tts"]
        base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

        # Build ordered list of directories to search for model files.
        _search_dirs = [
            _DATA_DIR,                          # ~/Library/Application Support/Jarvis/
            Path(__file__).parent,              # bundle Resources or project root (local dev)
        ]
        _bundle_dir = os.environ.get("JARVIS_BUNDLE_DIR")
        if _bundle_dir:
            _search_dirs.append(Path(_bundle_dir))  # folder containing the .app

        def _resolve(key: str, default: str) -> Path:
            """Search known dirs for the model file; fall back to _DATA_DIR as download target."""
            name = Path(c.get(key, default)).name
            for d in _search_dirs:
                p = d / name
                if p.is_file():
                    return p
            return _DATA_DIR / name   # download destination

        model_p  = _resolve("model_file",  "kokoro-v1.0.onnx")
        voices_p = _resolve("voices_file", "voices-v1.0.bin")
        if not model_p.is_file():
            _download(f"{base}/{model_p.name}", model_p)
        if not voices_p.is_file():
            _download(f"{base}/{voices_p.name}", voices_p)
        print(f"[TTS] Loading Kokoro ONNX  (voice: {c['voice']}) …")
        self._kokoro = Kokoro(str(model_p), str(voices_p))
        self._voice: str  = c["voice"]
        self._speed: float = float(c.get("speed", 1.0))
        print("[TTS] Ready")

    def _load_stt(self) -> None:
        from faster_whisper import WhisperModel

        c = self.cfg["stt"]
        # "model" wins over "model_size" — it accepts a plain size ("tiny", "base")
        # *or* a HuggingFace repo id such as "distil-whisper/distil-small.en".
        size = c.get("model") or c.get("model_size", "base")
        compute_type = c.get("compute_type", "int8")
        print(f"[STT] Loading faster-whisper '{size}' …")
        try:
            # Always try local cache first — avoids HuggingFace network call when offline.
            self._stt = WhisperModel(
                size, device="cpu", compute_type=compute_type, local_files_only=True
            )
        except Exception:
            # Model not in local cache yet — download it (requires internet).
            print(f"[STT] Model not cached — downloading faster-whisper '{size}' …")
            self._stt = WhisperModel(
                size, device="cpu", compute_type=compute_type, local_files_only=False
            )
        self._lang: str = c.get("language", "en")
        # beam_size 1 = greedy (fastest). 5 was the old default and costs ~2-3x
        # the decode time for a marginal WER gain on short command-style speech.
        self._beam_size: int = max(1, int(c.get("beam_size", STT_BEAM_SIZE_DEFAULT)))
        print(f"[STT] Ready (beam_size={self._beam_size})")

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _drain_q(self) -> None:
        """Discard stale frames left in the audio queue."""
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def record_audio(self, prefix: bytes = b"") -> bytes:
        """
        Record a full user utterance.
        Uses adaptive silence: short commands cut off at 600 ms,
        longer speech (> 2.5 s) gets 950 ms — so you can finish long sentences.

        `prefix` carries frames already captured elsewhere — the speech that
        triggered a barge-in — so the interrupting utterance keeps its opening
        word instead of starting mid-syllable.
        """
        while ws_server.is_muted():
            ws_server.set_state("idle")
            time.sleep(0.1)

        self._drain_q()
        ws_server.set_state("listening")
        print("🎤  Listening …", flush=True)
        buf        = prefix
        silence_ms = 0
        # Frames are 16-bit mono, so 2 bytes per sample.
        speech_ms  = len(prefix) * 1_000 // (2 * SAMPLE_RATE)
        speaking   = bool(prefix)

        def _cb(indata: np.ndarray, *_) -> None:
            self._audio_q.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            dtype="int16",
            channels=1,
            callback=_cb,
        ):
            while True:
                if ws_server.is_muted():
                    return b""
                frame = self._audio_q.get()
                if self.vad.is_speech(frame, SAMPLE_RATE):
                    buf       += frame
                    silence_ms = 0
                    speaking   = True
                    speech_ms += FRAME_MS
                elif speaking:
                    buf        += frame
                    silence_ms += FRAME_MS
                    cutoff = (
                        SILENCE_CUTOFF_LONG_MS
                        if speech_ms >= LONG_SPEECH_THRESHOLD_MS
                        else SILENCE_CUTOFF_SHORT_MS
                    )
                    if silence_ms > cutoff:
                        break
        return buf

    # ── STT ───────────────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes) -> str:
        audio = np.frombuffer(audio_bytes, dtype="int16").astype("float32") / 32_768.0
        segments, _ = self._stt.transcribe(
            audio,
            language=self._lang,
            beam_size=self._beam_size,
            temperature=0,                      # deterministic, no random sampling
            condition_on_previous_text=False,   # no hallucination from prior context
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,           # keep a bit of audio around speech edges
            },
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _synthesise(self, text: str) -> np.ndarray:
        samples, _ = self._kokoro.create(
            text, voice=self._voice, speed=self._speed, lang="en-us"
        )
        return np.asarray(samples, dtype=np.float32)

    def speak_direct(self, text: str, allow_barge_in: bool = True) -> None:
        """Speak text immediately via TTS — no LLM involved."""
        # Callers on other threads (the proactive monitor) queue behind the
        # current speaker instead of overlapping with it.
        with self._speak_lock:
            self._stop_speak.clear()
            ws_server.set_state("speaking")
            player   = SeamlessPlayer(sample_rate=TTS_RATE)
            listener = self._make_barge_listener(player) if allow_barge_in else None
            try:
                wav = self._synthesise(text)
                player.start()
                player.feed(wav)
                player.mark_done()
                if listener is not None:
                    listener.start()
                player.wait()
            finally:
                if listener is not None:
                    listener.stop()
                    self._consume_barge_in(listener)
                player.stop()
                ws_server.set_state("idle")

    def stop_speaking(self) -> None:
        self._stop_speak.set()

    # ── Barge-in ──────────────────────────────────────────────────────────────

    def _make_barge_listener(
        self, player: SeamlessPlayer,
        sentence_q: "Optional[queue.Queue[Optional[str]]]" = None,
    ) -> Optional[BargeInListener]:
        """
        Build a listener that kills playback the moment the user starts talking.

        Returns None when barge-in is disabled, in which case callers just skip
        start()/stop() — nothing else in the pipeline changes.
        """
        if not self._barge_in_enabled:
            return None

        def _on_trigger() -> None:
            self._stop_speak.set()
            player.stop()                       # cut the audio device immediately
            if sentence_q is not None:
                while True:                     # drop anything still queued for TTS
                    try:
                        sentence_q.get_nowait()
                    except queue.Empty:
                        break
                sentence_q.put(None)            # release a blocked TTS thread
            self._barge_in.set()
            print("\n✋  Interrupted — listening …", flush=True)

        return BargeInListener(_on_trigger, **self._barge_in_opts)

    def _consume_barge_in(self, listener: Optional[BargeInListener]) -> None:
        """Stash the audio that triggered the interrupt for the next recording."""
        if listener is not None and listener.triggered.is_set():
            self._barge_in_audio = listener.prefix

    def _take_barge_in_audio(self) -> bytes:
        """Pop the pending interrupt audio (empty when no barge-in happened)."""
        if not self._barge_in.is_set():
            return b""
        audio, self._barge_in_audio = self._barge_in_audio, b""
        self._barge_in.clear()
        return audio

    # ── System commands ───────────────────────────────────────────────────────

    # Spoken folder names → filesystem paths
    _FINDER_FOLDERS: dict[str, str] = {
        "downloads":    "~/Downloads",
        "download":     "~/Downloads",
        "desktop":      "~/Desktop",
        "documents":    "~/Documents",
        "document":     "~/Documents",
        "home":         "~",
        "pictures":     "~/Pictures",
        "picture":      "~/Pictures",
        "movies":       "~/Movies",
        "music":        "~/Music",
        "applications": "/Applications",
    }

    # Common spoken names → exact macOS .app names
    _APP_ALIASES: dict[str, str] = {
        "safari":               "Safari",
        "chrome":               "Google Chrome",
        "google chrome":        "Google Chrome",
        "firefox":              "Firefox",
        "spotify":              "Spotify",
        "discord":              "Discord",
        "slack":                "Slack",
        "whatsapp":             "WhatsApp",
        "telegram":             "Telegram",
        "notes":                "Notes",
        "calendar":             "Calendar",
        "finder":               "Finder",
        "terminal":             "Terminal",
        "xcode":                "Xcode",
        "vs code":              "Visual Studio Code",
        "vscode":               "Visual Studio Code",
        "visual studio code":   "Visual Studio Code",
        "cursor":               "Cursor",
        "mail":                 "Mail",
        "messages":             "Messages",
        "facetime":             "FaceTime",
        "maps":                 "Maps",
        "photos":               "Photos",
        "music":                "Music",
        "podcasts":             "Podcasts",
        "system preferences":   "System Preferences",
        "system settings":      "System Settings",
        "activity monitor":     "Activity Monitor",
        "calculator":           "Calculator",
        "preview":              "Preview",
        "arc":                  "Arc",
        "figma":                "Figma",
        "notion":               "Notion",
        "zoom":                 "Zoom",
        "ChatGPT":               "ChatGPT",
        "Claude":                "Claude",
    }

    def _resolve_app_name(self, raw: str) -> str:
        """Clean up transcription noise and map spoken names to exact app names."""
        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        clean = re.sub(r"^(?:the|a|an)\s+", "", clean)   # strip leading articles
        if clean in self._APP_ALIASES:
            return self._APP_ALIASES[clean]
        return clean.title()

    # Words that signal the captured text is NOT an app name
    _NON_APP_FIRST_WORDS = {
        "up", "down", "in", "out", "on", "off", "to", "with", "about",
        "for", "new", "my", "your", "this", "that", "some", "all", "more",
        "less", "much", "another", "any", "every", "it", "him", "her",
        "them", "us", "me", "both", "few", "many",
    }

    def _is_app_command(self, raw: str) -> bool:
        """
        Return True only if the captured text genuinely looks like an app name.
        Guards against false positives like 'open up about...' or 'close enough'.
        """
        clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
        clean = re.sub(r"^(?:the|a|an)\s+", "", clean)   # strip leading articles
        if clean in self._APP_ALIASES:
            return True
        words = clean.split()
        # Only allow 1–2 word names whose first word isn't a common non-app word
        return (
            1 <= len(words) <= 2
            and bool(words)
            and words[0] not in self._NON_APP_FIRST_WORDS
        )

    @staticmethod
    def _as_quote(value: str) -> str:
        """
        Escape a value for embedding inside an AppleScript double-quoted string.

        App names and labels here come from Whisper transcription, which happily
        emits quotes and backslashes. Interpolated raw, a name containing `"`
        closes the string literal early and the remainder is parsed as
        AppleScript — so anything spoken becomes something executed. Escaping
        backslash first, then the quote, keeps the value a value.
        """
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def _applescript(self, script: str) -> str:
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True
        )
        return result.stdout.strip()

    # ── Persistent memory ─────────────────────────────────────────────────────

    def _memory_set(self, key: str, value: str) -> None:
        self._mem.store(f"{key}: {value}", importance=0.7, tags=[key.lower().strip()])

    def _memory_recall(self, query: str = "") -> str:
        try:
            q = query or "recent memories"
            results = self._mem.recall(q, top_k=5)
            if not results:
                return "I don't have anything stored in memory yet, Sir."
            parts = [r.memory.content for r in results]
            return "I remember — " + "; ".join(parts) + "."
        except Exception:
            return "I couldn't access my memory right now, Sir."

    def _memory_forget(self, query: str) -> str:
        try:
            results = self._mem.recall(query, top_k=3)
            if not results:
                return f"I didn't find anything about {query} to forget, Sir."
            for r in results:
                self._mem.forget(r.memory.id)
            return f"Done, I've forgotten about {query}, Sir."
        except Exception:
            return "I couldn't update my memory right now, Sir."

    # ── Weather ───────────────────────────────────────────────────────────────

    _WMO: dict[int, str] = {
        0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "icy fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow",
        80: "rain showers", 81: "rain showers", 82: "violent rain showers",
        95: "thunderstorms", 96: "thunderstorms with hail", 99: "heavy thunderstorms",
    }

    def _get_weather(self, location: str = "") -> str:
        try:
            if not location:
                with urllib.request.urlopen("http://ip-api.com/json/", timeout=4) as r:
                    geo = json.loads(r.read())
                lat, lon, city = geo["lat"], geo["lon"], geo.get("city", "your area")
            else:
                geo_url = (
                    "https://geocoding-api.open-meteo.com/v1/search?name="
                    + urllib.parse.quote(location) + "&count=1&format=json"
                )
                with urllib.request.urlopen(geo_url, timeout=4) as r:
                    geo_data = json.loads(r.read())
                res  = geo_data["results"][0]
                lat, lon, city = res["latitude"], res["longitude"], res.get("name", location)

            w_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,apparent_temperature,weathercode,windspeed_10m"
                f"&temperature_unit=celsius&windspeed_unit=kmh&timezone=auto"
            )
            with urllib.request.urlopen(w_url, timeout=5) as r:
                w = json.loads(r.read())

            curr  = w["current"]
            temp  = round(curr["temperature_2m"])
            feels = round(curr["apparent_temperature"])
            wind  = round(curr["windspeed_10m"])
            desc  = self._WMO.get(int(curr["weathercode"]), "mixed conditions")
            return (
                f"Currently {desc} in {city}, {temp} degrees Celsius, "
                f"feels like {feels}, wind at {wind} kilometres per hour, Sir."
            )
        except Exception:
            return "I couldn't retrieve the weather right now, Sir."

    # ── Window management ─────────────────────────────────────────────────────

    def _window_maximize(self) -> str:
        # Click the zoom button of the frontmost window
        self._applescript(
            'tell application "System Events" to tell process '
            '(name of first application process whose frontmost is true) '
            'to perform action "AXZoom" of window 1'
        )
        return "Window maximized, Sir."

    def _window_hide_others(self) -> str:
        self._applescript(
            'tell application "System Events" to set visible of every process '
            'whose visible is true and frontmost is false to false'
        )
        return "Hiding all other windows, Sir. Focus mode activated."

    def _window_side_by_side(self, app1: str, app2: str) -> str:
        # Use macOS Split View via Mission Control shortcut
        name = self._as_quote(app1)
        self._applescript(f'''
        tell application "{name}" to activate
        delay 0.4
        tell application "System Events"
            tell process "{name}"
                set btn to button 3 of window 1
                perform action "AXShowMenu" of btn
            end tell
        end tell
        ''')
        return f"Attempting split view with {app1}, Sir."

    # ── Proactive monitor ─────────────────────────────────────────────────────

    def _check_battery(self) -> Optional[str]:
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
            m = re.search(r"(\d+)%", out)
            if m and "discharging" in out.lower():
                pct = int(m.group(1))
                if pct <= 20:
                    return f"Sir, your battery is at {pct} percent. I recommend plugging in."
        except Exception:
            pass
        return None

    def _check_calendar_soon(self) -> Optional[str]:
        try:
            result = self._applescript('''
                tell application "Calendar"
                    set nowDate to current date
                    set soonDate to nowDate + (10 * minutes)
                    set hits to {}
                    repeat with cal in every calendar
                        repeat with ev in every event of cal
                            try
                                if start date of ev >= nowDate and start date of ev <= soonDate then
                                    set end of hits to (summary of ev)
                                end if
                            end try
                        end repeat
                    end repeat
                    if (count of hits) > 0 then
                        return item 1 of hits
                    end if
                    return ""
                end tell
            ''')
            if result and result.strip():
                return f"Sir, you have an event coming up in the next 10 minutes: {result.strip()}."
        except Exception:
            pass
        return None

    def _morning_briefing(self) -> str:
        day     = time.strftime("%A, %B %-d")
        weather = self._get_weather()
        mem_str = ""
        try:
            results = self._mem.recall("reminder goal important", top_k=2)
            if results:
                entries = [r.memory.content for r in results]
                mem_str = " Also, a quick reminder: " + "; ".join(entries) + "."
        except Exception:
            pass
        return f"Good morning, Sir. Today is {day}. {weather}{mem_str} Have a great day."

    def _proactive_loop(self) -> None:
        """
        Background monitor. Every speak_direct() call from here contends for
        _speak_lock with the main loop, so an announcement waits its turn on the
        output device instead of overlapping with a reply already in progress.
        """
        _battery_alerted  = False
        _last_briefing_day = -1
        while True:
            time.sleep(60)
            try:
                now = time.localtime()
                # Morning briefing at 8:00 AM
                if now.tm_hour == 8 and now.tm_min < 2 and now.tm_mday != _last_briefing_day:
                    _last_briefing_day = now.tm_mday
                    self.speak_direct(self._morning_briefing(), allow_barge_in=False)

                # Battery alert
                batt_msg = self._check_battery()
                if batt_msg and not _battery_alerted:
                    _battery_alerted = True
                    self.speak_direct(batt_msg, allow_barge_in=False)
                elif not batt_msg:
                    _battery_alerted = False

                # Calendar (every 5 min)
                if now.tm_min % 5 == 0 and now.tm_sec < 65:
                    cal_msg = self._check_calendar_soon()
                    if cal_msg:
                        self.speak_direct(cal_msg, allow_barge_in=False)
            except Exception:
                pass

    # ── Clipboard helpers ─────────────────────────────────────────────────────

    # Action words that, combined with clipboard context, trigger clipboard mode
    _CLIPBOARD_ACTIONS = (
        "improve", "fix", "rewrite", "correct", "proofread", "summarize",
        "translate", "shorten", "lengthen", "shorter", "longer",
        "formal", "casual", "informal", "simplify", "explain", "clean up",
        "rephrase", "paraphrase", "polish", "edit", "check",
        # German
        "verbessere", "verbessern", "korrigiere", "korrigieren", "übersetze",
        "übersetzen", "kürze", "formuliere", "umschreiben", "prüfe",
    )

    # Explicit clipboard references — any of these alone triggers clipboard mode
    _CLIPBOARD_REFS = (
        "clipboard", "my clipboard", "the clipboard",
        "what i copied", "what i've copied", "the text i copied",
        "this text", "the text", "my text",
        # German
        "zwischenablage", "was ich kopiert habe", "den text",
    )

    def _try_augment_clipboard(self, text: str) -> tuple[str, bool]:
        """
        Detect clipboard commands and inject the clipboard content into the prompt.
        Also injects an instruction so the LLM returns ONLY the processed text
        (no preamble), making it safe to copy straight back to clipboard.
        Returns (augmented_text, is_clipboard_command).
        """
        t = text.lower().strip()

        has_ref    = any(ref in t for ref in self._CLIPBOARD_REFS)
        has_action = any(act in t for act in self._CLIPBOARD_ACTIONS)

        if not (has_ref or has_action):
            return text, False

        clipboard = subprocess.run(
            ["pbpaste"], capture_output=True, text=True
        ).stdout.strip()

        if not clipboard:
            # No clipboard content — still pass to LLM but don't copy back
            return text, False

        augmented = (
            f"{text}\n\n"
            f"Text to process:\n{clipboard}\n\n"
            f"IMPORTANT: Reply with ONLY the processed/improved text. "
            f"No preamble, no explanation, no quotes — just the result."
        )
        return augmented, True

    def _copy_to_clipboard(self, text: str) -> None:
        subprocess.run(["pbcopy"], input=text.encode(), check=False)

    def _timer_callback(self, seconds: int, label: str) -> None:
        time.sleep(seconds)
        msg = f"Sir, your {label} timer is up."
        print(f"\n⏰  {msg}", flush=True)
        subprocess.run(
            ["osascript", "-e",
             'display notification "Timer complete!" with title "Jarvis" '
             f'subtitle "{self._as_quote(label)}"'],
            check=False,
        )
        self.speak_direct(msg)

    def _handle_system_command(self, text: str) -> Optional[str]:
        """
        Check whether `text` is a local system command.
        If yes: execute it and return the spoken response string.
        If no:  return None  (caller should send to LLM).
        """
        t = text.lower().strip()

        # ── Date & time ───────────────────────────────────────────────────────
        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?time|what\s+time\s+is\s+it)\b", t):
            now = time.strftime("%-I:%M %p")
            return f"It's {now}, Sir."

        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:today'?s?\s+)?date|what(?:'s|\s+is)\s+today|today'?s?\s+date)\b", t):
            today = time.strftime("%A, %B %-d")
            return f"Today is {today}, Sir."

        # ── System info ───────────────────────────────────────────────────────
        if re.search(r"\b(?:how\s+much\s+(?:ram|memory)|(?:free|available)\s+(?:ram|memory)|memory\s+(?:usage|left|free))\b", t):
            try:
                vm      = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
                ps_m    = re.search(r"page size of (\d+) bytes", vm)
                page_sz = int(ps_m.group(1)) if ps_m else 16_384
                free    = int(re.search(r"Pages free:\s+(\d+)", vm).group(1))
                inact   = int(re.search(r"Pages inactive:\s+(\d+)", vm).group(1))
                avail   = round((free + inact) * page_sz / 1024 ** 3, 1)
                return f"About {avail} gigabytes of memory available, Sir."
            except Exception:
                return "I couldn't read the memory stats right now, Sir."

        if re.search(r"\b(?:cpu\s+usage|processor\s+(?:usage|load)|how\s+(?:busy|loaded)\s+(?:is\s+)?(?:the\s+)?cpu)\b", t):
            try:
                top = subprocess.run(
                    ["top", "-l", "1", "-n", "0", "-s", "0"],
                    capture_output=True, text=True, timeout=6,
                ).stdout
                m2 = re.search(r"CPU usage:\s+([\d.]+)%\s+user,\s+([\d.]+)%\s+sys", top)
                if m2:
                    used = round(float(m2.group(1)) + float(m2.group(2)), 1)
                    return f"CPU is at {used} percent usage right now, Sir."
            except Exception:
                pass
            return "I couldn't read the CPU stats right now, Sir."

        if re.search(r"\b(?:how\s+much\s+(?:storage|disk|space)|(?:storage|disk)\s+(?:space\s+)?(?:left|free|remaining|available)|free\s+(?:storage|disk|space))\b", t):
            try:
                df    = subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout.splitlines()
                parts = df[1].split()
                avail, pct = parts[3], parts[4]
                return f"{avail} of storage available, {pct} used, Sir."
            except Exception:
                return "I couldn't read the disk stats right now, Sir."

        # ── Volume query ──────────────────────────────────────────────────────
        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?volume|current\s+volume)\b", t):
            vol   = self._applescript("output volume of (get volume settings)")
            muted = self._applescript("output muted of (get volume settings)")
            if muted == "true":
                return "The volume is currently muted, Sir."
            return f"The volume is at {vol} percent, Sir."

        # ── Active app ────────────────────────────────────────────────────────
        if re.search(r"\b(?:what\s+am\s+i\s+(?:working\s+on|doing)|current(?:ly\s+(?:using|in|on))?|active\s+(?:app|window)|what(?:'s|\s+is)\s+(?:open|active|running|in\s+front))\b", t):
            app = self._applescript(
                'tell application "System Events" to get name of first application process whose frontmost is true'
            )
            return f"You're in {app} right now, Sir."

        # ── Maps navigation ───────────────────────────────────────────────────
        m = re.search(
            r"\b(?:navigate|directions?|route|take me|get me|show me the way)\s+"
            r"(?:me\s+)?(?:to|towards?)\s+(.+)",
            t,
        )
        if not m:
            m = re.search(r"\bhow\s+(?:do\s+i\s+get|can\s+i\s+get|to\s+get)\s+to\s+(.+)", t)
        if m:
            raw_dest = re.sub(r"[?.!,]+$", "", m.group(1).strip())
            encoded  = urllib.parse.quote(raw_dest)
            subprocess.run(["open", f"maps://?daddr={encoded}"], check=False)
            return f"Opening Maps with directions to {raw_dest}, Sir."

        # ── Finder folders ────────────────────────────────────────────────────
        if re.match(r"^open\s+", t):
            folder_key = re.sub(r"^open\s+", "", t).rstrip("., ").lower()
            if folder_key in self._FINDER_FOLDERS:
                subprocess.run(["open", self._FINDER_FOLDERS[folder_key]], check=False)
                return f"Opening your {folder_key.title()} folder, Sir."

        # ── Volume ────────────────────────────────────────────────────────────
        m = re.search(r"\bvolume\s+(?:to\s+)?(\d{1,3})\b", t)
        if m:
            vol = min(100, max(0, int(m.group(1))))
            self._applescript(f"set volume output volume {vol}")
            return f"Volume set to {vol} percent, Sir."

        if re.search(r"\bunmute\b", t):
            self._applescript("set volume output muted false")
            return "Unmuted, Sir."

        if re.search(r"\b(?:mute|silence)\b", t):
            self._applescript("set volume output muted true")
            return "Muted, Sir."

        if re.search(r"\b(?:turn\s+up|louder|raise\s+(?:the\s+)?volume|increase\s+(?:the\s+)?volume|volume\s+up)\b", t):
            cur = self._applescript("output volume of (get volume settings)")
            new_vol = min(100, int(cur or 50) + 15)
            self._applescript(f"set volume output volume {new_vol}")
            return f"Volume at {new_vol} percent."

        if re.search(r"\b(?:turn\s+down|quieter|lower\s+(?:the\s+)?volume|decrease\s+(?:the\s+)?volume|volume\s+down)\b", t):
            cur = self._applescript("output volume of (get volume settings)")
            new_vol = max(0, int(cur or 50) - 15)
            self._applescript(f"set volume output volume {new_vol}")
            return f"Volume at {new_vol} percent."

        # ── Screenshot ────────────────────────────────────────────────────────
        if re.search(r"\b(?:take|capture|make)\s+(?:a\s+)?screenshot\b", t):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = Path.home() / "Desktop" / f"screenshot_{ts}.png"
            subprocess.run(["screencapture", "-x", str(path)], check=False)
            return "Screenshot saved to your Desktop, Sir."

        # ── Timer ─────────────────────────────────────────────────────────────
        m = re.search(
            r"\b(?:set\s+(?:a\s+)?)?timer\s+(?:for\s+)?(\d+)\s*(second|minute|hour)s?\b", t
        )
        if m:
            amount  = int(m.group(1))
            unit    = m.group(2)
            seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
            label   = f"{amount} {unit}{'s' if amount != 1 else ''}"
            threading.Thread(
                target=self._timer_callback, args=(seconds, label), daemon=True
            ).start()
            return f"Timer set for {label}, Sir."

        # ── Reminder ──────────────────────────────────────────────────────────
        m = re.search(
            r"\bremind\s+me\s+in\s+(\d+)\s*(second|minute|hour)s?\b", t
        )
        if m:
            amount  = int(m.group(1))
            unit    = m.group(2)
            seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
            label   = f"{amount} {unit}{'s' if amount != 1 else ''}"
            threading.Thread(
                target=self._timer_callback, args=(seconds, label), daemon=True
            ).start()
            return f"I'll remind you in {label}, Sir."

        # ── Open app ──────────────────────────────────────────────────────────
        m = re.search(
            r"^(?:open|launch|start)\s+(.+?)(?:\s+(?:app|application))?\s*$", t
        )
        if m and self._is_app_command(m.group(1)):
            app_name = self._resolve_app_name(m.group(1))
            res = subprocess.run(["open", "-a", app_name], capture_output=True)
            if res.returncode == 0:
                return f"Opening {app_name}, Sir."
            return f"I couldn't find an app called {app_name}, Sir."

        # ── Quit app ──────────────────────────────────────────────────────────
        m = re.search(
            r"^(?:quit|close|exit|kill)\s+(.+?)(?:\s+(?:app|application))?\s*$", t
        )
        if m and self._is_app_command(m.group(1)):
            app_name = self._resolve_app_name(m.group(1))
            self._applescript(
                f'tell application "{self._as_quote(app_name)}" to quit'
            )
            return f"Closing {app_name}."

        # ── Orb demo ──────────────────────────────────────────────────────────
        if re.search(
            r"\b(?:show\s+me\s+(?:something|some(?:thing)?\s+cool(?:\s+thing)?s?|"
            r"what\s+you\s+can\s+do|your\s+moves?|off)|"
            r"do\s+something\s+cool|impress\s+me|show\s+off|"
            r"activate\s+(?:demo|show|display)|party\s+mode)\b",
            t,
        ):
            ws_server.send_event({"action": "demo"})
            return "Watch this, Sir."

        # ── Weather ───────────────────────────────────────────────────────────
        if re.search(r"\b(?:weather|forecast|temperature|how\s+(?:hot|cold|warm)\s+is\s+it|"
                     r"will\s+it\s+rain|is\s+it\s+(?:raining|snowing|sunny|cloudy))\b", t):
            m = re.search(r"\bin\s+([a-z\s]+?)(?:\s+(?:today|tomorrow|now|right now))?\s*[?.]?\s*$", t)
            loc = m.group(1).strip() if m else ""
            return self._get_weather(loc)

        # ── Persistent memory ─────────────────────────────────────────────────
        # "remember that X is Y" / "remember: X"
        m = re.search(
            r"\b(?:remember\s+(?:that\s+)?|note\s+(?:that\s+)?|save\s+(?:that\s+)?)(.+)", t
        )
        if m:
            fact = m.group(1).strip().rstrip(".,!")
            # Try to split "X is Y" or "X: Y"
            kv = re.split(r"\s+is\s+|\s*:\s*", fact, maxsplit=1)
            if len(kv) == 2:
                self._memory_set(kv[0], kv[1])
                return f"Got it, Sir. I'll remember that {kv[0]} is {kv[1]}."
            else:
                self._memory_set(fact, fact)
                return f"Noted, Sir: {fact}."

        # "what do you remember" / "recall X"
        m = re.search(r"\b(?:what\s+do\s+you\s+remember|recall|remember\s+about|what\s+did\s+i\s+tell\s+you)\b.*?(?:about\s+(.+))?$", t)
        if m:
            query = (m.group(1) or "").strip().rstrip("?.!")
            return self._memory_recall(query)

        # "forget about X"
        m = re.search(r"\bforget\s+(?:about\s+)?(.+)", t)
        if m:
            return self._memory_forget(m.group(1).strip().rstrip("?.!"))

        # ── Window management ─────────────────────────────────────────────────
        if re.search(r"\b(?:maximize|full\s*screen|make\s+(?:the\s+)?window\s+(?:bigger|larger|fullscreen))\b", t):
            return self._window_maximize()

        if re.search(r"\b(?:focus\s+mode|hide\s+(?:all\s+)?other(?:s|\s+windows?)|"
                     r"show\s+only\s+this|distraction\s+free)\b", t):
            return self._window_hide_others()

        return None

    # ── Spinner ───────────────────────────────────────────────────────────────

    @staticmethod
    def _spinner(stop: threading.Event) -> None:
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not stop.is_set():
            sys.stdout.write(f"\r  Thinking {frames[i % len(frames)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()

    # ── Turn (LLM pipeline) ───────────────────────────────────────────────────

    def _messages(self) -> list[dict]:
        max_pairs = self._llm_cfg.get("history_turns", 10)
        recent    = self.history[-(max_pairs * 2):]
        return [{"role": "system", "content": self.system_prompt}] + recent

    def stream_sentences(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})

        stream = self._llm.create_chat_completion(
            messages=self._messages(),
            max_tokens=self._llm_cfg.get("max_new_tokens", 256),
            temperature=self._llm_cfg.get("temperature", 0.7),
            top_p=self._llm_cfg.get("top_p", 0.9),
            stop=["<|eot_id|>", "\nUser:", "\nYou:"],
            stream=True,
        )

        buf  = ""
        full = ""

        for chunk in stream:
            # A barge-in already killed playback — stop burning tokens on a
            # reply nobody is going to hear.
            if self._stop_speak.is_set():
                break
            delta: str = chunk["choices"][0]["delta"].get("content", "") or ""
            buf  += delta
            full += delta

            parts = SENTENCE_RE.split(buf)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    c = _clean(sentence)
                    if c:
                        yield c
                buf = parts[-1]
                continue

            if len(buf.split()) >= MIN_CLAUSE_WORDS:
                clauses = CLAUSE_RE.split(buf)
                if len(clauses) > 1:
                    for clause in clauses[:-1]:
                        c = _clean(clause)
                        if c:
                            yield c
                    buf = clauses[-1]

        if buf.strip():
            c = _clean(buf)
            if c:
                yield c

        self.history.append({"role": "assistant", "content": _clean(full)})

    def handle_turn(self, user_input: str) -> None:
        """Three-thread pipeline: LLM → TTS → SeamlessPlayer (zero-gap audio)."""
        # One turn owns the output device from first sentence to last, so a
        # proactive announcement can't start mid-reply.
        with self._speak_lock:
            self._stop_speak.clear()
            ws_server.set_state("thinking")

            sentence_q: queue.Queue[Optional[str]] = queue.Queue()
            player = SeamlessPlayer(sample_rate=TTS_RATE)
            player.start()
            listener = self._make_barge_listener(player, sentence_q)

            first_audio_ready = threading.Event()
            display_parts: list[str] = []
            display_lock = threading.Lock()

            def _llm() -> None:
                for chunk in self.stream_sentences(user_input):
                    sentence_q.put(chunk)
                sentence_q.put(None)

            def _tts() -> None:
                first = True
                while True:
                    chunk = sentence_q.get()
                    if chunk is None:
                        break
                    if self._stop_speak.is_set():
                        break
                    wav = self._synthesise(chunk)
                    player.feed(wav)
                    with display_lock:
                        display_parts.append(chunk)
                    if first:
                        ws_server.set_state("speaking")
                        first_audio_ready.set()
                        first = False
                player.mark_done()

            llm_t = threading.Thread(target=_llm, daemon=True)
            tts_t = threading.Thread(target=_tts, daemon=True)

            stop_spin = threading.Event()
            spin_t    = threading.Thread(
                target=self._spinner, args=(stop_spin,), daemon=True
            )
            spin_t.start()
            llm_t.start()
            tts_t.start()

            got_audio = first_audio_ready.wait(timeout=60)
            stop_spin.set()
            spin_t.join()

            # Only listen for an interrupt once audio is actually coming out.
            if got_audio and listener is not None:
                listener.start()

            tts_t.join()
            with display_lock:
                response_text = " ".join(display_parts)
            sys.stdout.write(f"Jarvis: {response_text}\n")
            sys.stdout.flush()

            player.wait()
            if listener is not None:
                listener.stop()
                self._consume_barge_in(listener)
            # stream_sentences checks _stop_speak, so an interrupted generation
            # unwinds promptly; the timeout is a backstop, not the normal path.
            llm_t.join(timeout=5.0)
            ws_server.set_state("idle")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "═" * 58)
        print("  🟢  Voice assistant ready — just speak!")
        if self._wake is not None:
            print(f"  Wake word active after {self._wake_timeout:.0f}s silence — say 'Hey Jarvis'")
        else:
            print("  Wake word disabled — always listening")
        print("  Open http://localhost:3000 to see the UI")
        print("  Press Ctrl+C to quit")
        print("═" * 58 + "\n")

        _last_input = time.time()
        _wake_mode  = False
        # Speech captured by a barge-in, replayed into the next recording.
        _prefix     = b""

        while True:
            try:
                if ws_server.is_muted():
                    ws_server.set_state("idle")
                    time.sleep(0.1)
                    continue

                # Switch to wake-word mode after timeout
                if (self._wake is not None and not _wake_mode
                        and time.time() - _last_input > self._wake_timeout):
                    _wake_mode = True
                    print("💤  Wake-word mode — say 'Hey Jarvis' to wake me up.", flush=True)
                    ws_server.set_state("idle")

                if _wake_mode and not _prefix:
                    # Blocks on the wake-word model until it hears the phrase —
                    # no Whisper, no LLM, near-zero cost while idle.
                    if not self._wake.listen(should_abort=ws_server.is_muted):
                        continue
                    _wake_mode  = False
                    _last_input = time.time()
                    print("🟢  Woke up!", flush=True)
                    self.speak_direct("Yes, Sir?")
                    continue

                audio  = self.record_audio(prefix=_prefix)
                _prefix = b""
                if not audio:
                    continue

            except KeyboardInterrupt:
                print("\nGoodbye, Sir.")
                ws_server.set_state("idle")
                break

            user_input = self.transcribe(audio)
            if not user_input:
                print("  (Didn't catch that — try again)\n")
                continue

            _last_input = time.time()   # reset idle timer on every real input
            print(f"You: {user_input}")

            # Clipboard augmentation (before system-command check)
            augmented_input, is_clipboard = self._try_augment_clipboard(user_input)

            # System command (direct execution) or LLM
            sys_response = self._handle_system_command(user_input)
            if sys_response:
                print(f"System: {sys_response}")
                self.speak_direct(sys_response)
            else:
                self.handle_turn(augmented_input)
                # Copy LLM response back to clipboard when requested
                if is_clipboard and self.history:
                    last = self.history[-1].get("content", "").strip()
                    if last:
                        self._copy_to_clipboard(last)
                        print("📋  Improved text copied to clipboard.", flush=True)

            # The user talked over Jarvis: go straight back to recording with
            # the speech that triggered the interrupt, no "Yes, Sir?" roundtrip.
            _prefix = self._take_barge_in_audio()
            if _prefix:
                _wake_mode  = False
                _last_input = time.time()

            print()


if __name__ == "__main__":
    # HTTP/WebSocket sofort — bevor schwere ML-Imports in VoiceAssistant laufen.
    ws_server.start()
    assistant = VoiceAssistant()
    assistant.run()
