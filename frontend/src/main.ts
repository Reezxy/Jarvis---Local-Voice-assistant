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

// ── Orb ───────────────────────────────────────────────────────────────────────
const orb = createOrb(canvas);

// ── State labels ──────────────────────────────────────────────────────────────
const STATE_LABELS: Record<OrbState, string> = {
  idle: "",
  listening: "listening...",
  thinking: "thinking...",
  speaking: "",
};

function applyState(state: OrbState): void {
  orb.setState(state);
  statusEl.textContent = STATE_LABELS[state];
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
    const data = (await res.json()) as { state?: string; muted?: boolean };
    if (data.state) {
      applyState(data.state as OrbState);
    }
    if (typeof data.muted === "boolean") {
      setMuted(data.muted);
    }
  } catch {
    // backend offline, keep UI defaults
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
void refreshStatus();
connect();
muteButtonEl.addEventListener("click", () => {
  void toggleMuted();
});

// Silence unused-import warning for showError — will be useful for future extensions
void showError;
