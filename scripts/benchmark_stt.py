#!/usr/bin/env python3
"""
Benchmark faster-whisper model size / beam_size against latency and accuracy.

Usage
─────
  # Use your own clips: samples/foo.wav + samples/foo.txt (reference transcript)
  python scripts/benchmark_stt.py --samples samples/

  # No clips? Synthesise them with the project's Kokoro TTS (reference = known)
  python scripts/benchmark_stt.py --synthesise

  # Pick what to sweep
  python scripts/benchmark_stt.py --synthesise \
      --models tiny.en base.en small distil-whisper/distil-small.en \
      --beams 1 2 5

Reports, per (model, beam_size): mean latency per clip, real-time factor,
and word error rate against the reference transcripts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16_000

# Short, command-shaped utterances — the traffic this assistant actually sees.
DEFAULT_PHRASES = [
    "Hey Jarvis, what's the weather like today?",
    "Open Safari and turn the volume up to sixty percent.",
    "Set a timer for fifteen minutes called laundry.",
    "Remind me that my flight to Berlin leaves at seven in the morning.",
    "Improve the text I just copied to the clipboard.",
    "Quit Spotify.",
    "What did I ask you about the project yesterday?",
    "Take a screenshot and put the windows side by side.",
]


def _normalise(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance over word tokens."""
    ref, hyp = _normalise(reference), _normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE:
        return audio
    n = int(round(len(audio) * SAMPLE_RATE / src_rate))
    return np.interp(
        np.linspace(0.0, len(audio) - 1, n), np.arange(len(audio)), audio
    ).astype(np.float32)


def load_samples(directory: Path) -> list[tuple[str, np.ndarray]]:
    """Load *.wav clips paired with a same-named *.txt reference transcript."""
    samples: list[tuple[str, np.ndarray]] = []
    for wav_path in sorted(directory.glob("*.wav")):
        ref_path = wav_path.with_suffix(".txt")
        if not ref_path.is_file():
            print(f"  ! skipping {wav_path.name} — no {ref_path.name} reference")
            continue
        with wave.open(str(wav_path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                print(f"  ! skipping {wav_path.name} — need 16-bit mono PCM")
                continue
            raw = w.readframes(w.getnframes())
            rate = w.getframerate()
        audio = np.frombuffer(raw, dtype="int16").astype("float32") / 32_768.0
        samples.append((ref_path.read_text(encoding="utf-8").strip(),
                        _resample_to_16k(audio, rate)))
    return samples


def synthesise_samples(phrases: list[str]) -> list[tuple[str, np.ndarray]]:
    """Render reference phrases with the project's Kokoro TTS voice."""
    from kokoro_onnx import Kokoro

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    tts = cfg.get("tts", {})
    kokoro = Kokoro(
        str(ROOT / tts.get("model_file", "kokoro-v1.0.onnx")),
        str(ROOT / tts.get("voices_file", "voices-v1.0.bin")),
    )
    samples = []
    for phrase in phrases:
        audio, rate = kokoro.create(
            phrase, voice=tts.get("voice", "am_fenrir"),
            speed=float(tts.get("speed", 1.0)), lang="en-us",
        )
        samples.append((phrase, _resample_to_16k(np.asarray(audio, "float32"), rate)))
    return samples


def benchmark(model_name: str, beams: list[int], samples, compute_type: str, language: str):
    from faster_whisper import WhisperModel

    print(f"\n[{model_name}] loading …", flush=True)
    load_start = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    load_s = time.perf_counter() - load_start

    rows = []
    for beam in beams:
        latencies, wers = [], []
        for reference, audio in samples:
            start = time.perf_counter()
            segments, _ = model.transcribe(
                audio,
                language=language,
                beam_size=beam,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            latencies.append(time.perf_counter() - start)
            wers.append(_wer(reference, text))
        audio_s = sum(len(a) for _, a in samples) / SAMPLE_RATE
        rows.append({
            "model": model_name,
            "beam_size": beam,
            "mean_latency_s": sum(latencies) / len(latencies),
            "rtf": sum(latencies) / audio_s,
            "wer": sum(wers) / len(wers),
            "load_s": load_s,
        })
        print(f"  beam={beam}  {rows[-1]['mean_latency_s']:.3f} s/clip  "
              f"RTF {rows[-1]['rtf']:.3f}  WER {rows[-1]['wer'] * 100:.1f}%", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=Path,
                    help="directory of 16-bit mono *.wav clips + *.txt references")
    ap.add_argument("--synthesise", action="store_true",
                    help="generate clips with the project's Kokoro TTS instead")
    ap.add_argument("--models", nargs="+",
                    default=["tiny.en", "base.en", "small",
                             "distil-whisper/distil-small.en"])
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 2, 5])
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--language", default="en")
    ap.add_argument("--json", type=Path, help="also write raw results here")
    args = ap.parse_args()

    if args.samples:
        samples = load_samples(args.samples)
    elif args.synthesise:
        samples = synthesise_samples(DEFAULT_PHRASES)
    else:
        ap.error("pass --samples DIR or --synthesise")

    if not samples:
        print("No usable samples found.")
        return 1
    print(f"{len(samples)} clip(s), "
          f"{sum(len(a) for _, a in samples) / SAMPLE_RATE:.1f} s of audio")

    rows = []
    for model_name in args.models:
        try:
            rows += benchmark(model_name, args.beams, samples,
                              args.compute_type, args.language)
        except Exception as exc:            # noqa: BLE001 — one bad model shouldn't abort the sweep
            print(f"  ! {model_name} failed: {exc}")

    if not rows:
        return 1

    print("\n" + "─" * 74)
    print(f"{'model':<34}{'beam':>5}{'s/clip':>10}{'RTF':>8}{'WER':>9}")
    print("─" * 74)
    for r in sorted(rows, key=lambda r: r["mean_latency_s"]):
        print(f"{r['model']:<34}{r['beam_size']:>5}{r['mean_latency_s']:>10.3f}"
              f"{r['rtf']:>8.3f}{r['wer'] * 100:>8.1f}%")
    print("─" * 74)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
