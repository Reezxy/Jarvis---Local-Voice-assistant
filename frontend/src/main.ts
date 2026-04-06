/**
 * Voice Assistant UI — main entry point.
 *
 * Connects to the Python WebSocket bridge (ws://localhost:8765),
 * receives state updates and drives the Three.js orb accordingly.
 *
 * States: "idle" | "listening" | "thinking" | "speaking"
 */

import { createOrb, type OrbState } from "./orb";
import "./style.css";

// ── Config ────────────────────────────────────────────────────────────────────
const WS_URL = "ws://localhost:8765";
const RECONNECT_INTERVAL_MS = 2_000;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const statusEl = document.getElementById("status-text") as HTMLDivElement;
const errorEl = document.getElementById("error-text") as HTMLDivElement;
const badgeEl = document.getElementById("connection-badge") as HTMLDivElement;
const badgeLabelEl = document.getElementById(
  "connection-label"
) as HTMLSpanElement;
const muteButtonEl = document.getElementById("mute-button") as HTMLButtonElement;
const langButtonEl = document.getElementById("lang-button") as HTMLButtonElement;
const langFlagEl   = document.getElementById("lang-flag")   as HTMLSpanElement;
const langLabelEl  = document.getElementById("lang-label")  as HTMLSpanElement;

// ── Orb ───────────────────────────────────────────────────────────────────────
const orb = createOrb(canvas);

// ── Language ──────────────────────────────────────────────────────────────────
let currentLang: "en" | "de" = "en";

const STATE_LABELS: Record<"en" | "de", Record<OrbState, string>> = {
  en: { idle: "", listening: "listening...", thinking: "thinking...", speaking: "" },
  de: { idle: "", listening: "zuhören...", thinking: "denken...", speaking: "" },
};

function setLang(lang: "en" | "de"): void {
  currentLang = lang;
  langButtonEl.setAttribute("data-lang", lang);
  if (lang === "de") {
    langFlagEl.textContent  = "🇩🇪";
    langLabelEl.textContent = "DE";
  } else {
    langFlagEl.textContent  = "🇺🇸";
    langLabelEl.textContent = "EN";
  }
}

function applyState(state: OrbState): void {
  orb.setState(state);
  statusEl.textContent = STATE_LABELS[currentLang][state];
}

function setMuted(muted: boolean): void {
  muteButtonEl.classList.toggle("is-muted", muted);
  muteButtonEl.setAttribute("aria-pressed", String(muted));
  muteButtonEl.textContent = muted ? "unmute" : "mute";
}

// ── Error toast ───────────────────────────────────────────────────────────────
let errorTimer: ReturnType<typeof setTimeout> | null = null;

function showError(msg: string): void {
  errorEl.textContent = msg;
  errorEl.style.opacity = "1";
  if (errorTimer) clearTimeout(errorTimer);
  errorTimer = setTimeout(() => {
    errorEl.style.opacity = "0";
  }, 4_000);
}

// ── Connection badge ──────────────────────────────────────────────────────────
function setConnected(ok: boolean): void {
  badgeEl.classList.toggle("connected", ok);
  badgeEl.classList.toggle("disconnected", !ok);
  badgeLabelEl.textContent = ok ? "connected" : "reconnecting";
  muteButtonEl.disabled = !ok;
}

async function refreshStatus(): Promise<void> {
  try {
    const res = await fetch("http://localhost:3000/api/status");
    if (!res.ok) return;
    const data = (await res.json()) as { state?: string; muted?: boolean; lang?: string };
    if (data.state) {
      applyState(data.state as OrbState);
    }
    if (typeof data.muted === "boolean") {
      setMuted(data.muted);
    }
    if (data.lang === "en" || data.lang === "de") {
      setLang(data.lang);
    }
  } catch {
    // backend offline, keep UI defaults
  }
}

async function toggleLang(): Promise<void> {
  const nextLang = currentLang === "en" ? "de" : "en";
  try {
    const res = await fetch("http://localhost:3000/api/lang", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang: nextLang }),
    });
    if (!res.ok) throw new Error("lang request failed");
    const data = (await res.json()) as { lang?: string };
    if (data.lang === "en" || data.lang === "de") {
      setLang(data.lang);
    }
  } catch {
    showError("language switch failed");
  }
}

async function toggleMuted(): Promise<void> {
  const nextMuted = muteButtonEl.getAttribute("aria-pressed") !== "true";
  try {
    const res = await fetch("http://localhost:3000/api/mute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ muted: nextMuted }),
    });
    if (!res.ok) {
      throw new Error("mute request failed");
    }
    const data = (await res.json()) as { muted?: boolean; state?: string };
    if (typeof data.muted === "boolean") {
      setMuted(data.muted);
    }
    if (data.state) {
      applyState(data.state as OrbState);
    }
  } catch {
    showError("mute toggle failed");
  }
}

// ── WebSocket with auto-reconnect ─────────────────────────────────────────────
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function connect(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  ws = new WebSocket(WS_URL);

  ws.addEventListener("open", () => {
    setConnected(true);
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data as string) as {
        state?: string;
        muted?: boolean;
        lang?: string;
        action?: string;
      };
      if (data.action === "demo") {
        orb.triggerDemo();
        return;
      }
      if (data.state) {
        applyState(data.state as OrbState);
      }
      if (typeof data.muted === "boolean") {
        setMuted(data.muted);
      }
      if (data.lang === "en" || data.lang === "de") {
        setLang(data.lang);
      }
    } catch {
      // ignore malformed messages
    }
  });

  ws.addEventListener("close", () => {
    setConnected(false);
    applyState("idle");
    scheduleReconnect();
  });

  ws.addEventListener("error", () => {
    // error is always followed by close — handled there
    setConnected(false);
  });
}

function scheduleReconnect(): void {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, RECONNECT_INTERVAL_MS);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
setConnected(false);
applyState("idle");
setMuted(false);
setLang("en");
void refreshStatus();
connect();
muteButtonEl.addEventListener("click", () => { void toggleMuted(); });
langButtonEl.addEventListener("click", () => { void toggleLang(); });
