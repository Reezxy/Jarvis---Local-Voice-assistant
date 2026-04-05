#!/bin/bash
# ══════════════════════════════════════════════════════
#   VOICE ASSISTANT — macOS Launcher (100 % offline)
#   Doppelklick im Finder → startet alles automatisch
#   Kein Internet nötig nach dem ersten Build.
#
#   SCRIPT_REV=2026-04-04  (Chrome, dist-Build, Port 3000/8765)
#   Wenn du hier eine andere Rev. siehst: falscher Ordner / alte Kopie.
# ══════════════════════════════════════════════════════

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

clear
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         🎙  VOICE ASSISTANT              ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  → Launcher: $DIR/start.command  (SCRIPT_REV=2026-04-04)"
echo ""

# ── Fehlerprüfung: Python-Umgebung ───────────────────
if [ ! -d "$DIR/.venv311" ]; then
  echo "  ✗  .venv311 nicht gefunden."
  echo "     Einmalig ausführen:"
  echo "     python3.11 -m venv .venv311"
  echo "     source .venv311/bin/activate"
  echo "     pip install -r requirements_speech_to_speech.txt"
  echo ""
  read -rp "  Enter zum Beenden…"
  exit 1
fi

# ── Frontend einmalig bauen (nur wenn dist/ fehlt) ────
if [ ! -d "$DIR/frontend/dist" ]; then
  echo "  ▶  Erstelle UI-Build (einmalig, ~30 Sek.)…"
  echo "     (danach startet alles ohne Node.js / Internet)"
  echo ""

  if ! command -v npm &>/dev/null; then
    echo "  ✗  npm nicht gefunden. Node.js installieren:"
    echo "     https://nodejs.org  (nur für diesen einmaligen Build nötig)"
    echo ""
    read -rp "  Enter zum Beenden…"
    exit 1
  fi

  cd "$DIR/frontend"
  [ ! -d "node_modules" ] && npm install
  npm run build
  cd "$DIR"

  if [ ! -d "$DIR/frontend/dist" ]; then
    echo "  ✗  Build fehlgeschlagen."
    read -rp "  Enter zum Beenden…"
    exit 1
  fi
  echo ""
  echo "  ✅  UI gebaut — ab jetzt kein Node.js mehr nötig!"
  echo ""
fi

# ── Aufräumen beim Beenden ────────────────────────────
PYTHON_PID=""

cleanup() {
  echo ""
  echo "  Stopping Voice Assistant…"
  [ -n "$PYTHON_PID" ] && kill "$PYTHON_PID" 2>/dev/null
  lsof -ti :8765 | xargs kill -9 2>/dev/null
  lsof -ti :3000  | xargs kill -9 2>/dev/null
  echo "  Stopped. Tschüss!"
  exit 0
}
trap cleanup EXIT INT TERM

# ── Python starten (WebSocket + HTTP-Server + KI) ─────
echo "  ▶  Lade KI-Modelle  (~15 Sek.)…"
source "$DIR/.venv311/bin/activate"

python "$DIR/chatbot_speech_to_speech.py" &
PYTHON_PID=$!

# ── Warten bis HTTP-Server auf Port 3000 bereit ist ───
echo "  ▶  Starte UI-Server…"
for i in $(seq 1 30); do
  if lsof -i :3000 >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# ── Browser öffnen (Chrome) ──────────────────────────
if [ -d "/Applications/Google Chrome.app" ]; then
  open -a "Google Chrome" http://localhost:3000
else
  open http://localhost:3000
fi
echo ""
echo "  ══════════════════════════════════════════"
echo "  ✅  Voice Assistant läuft!"
echo "  ✅  Browser → http://localhost:3000"
echo "  ══════════════════════════════════════════"
echo ""
echo "  100 % offline — kein Internet benötigt."
echo "  Einfach sprechen — kein Tastendruck nötig."
echo "  Fenster schließen oder Ctrl+C zum Beenden."
echo ""

wait "$PYTHON_PID"
