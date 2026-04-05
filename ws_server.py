"""
Minimal WebSocket state-broadcast server + static HTTP server.

Runs two background threads alongside the voice assistant:
  • WebSocket on port 8765 – broadcasts state changes to the browser
  • HTTP     on port 3000  – serves the pre-built frontend (frontend/dist/)

No Node.js / npm required at runtime; build once with `npm run build`.

Usage inside Python:
    import ws_server
    ws_server.start()          # call once at startup
    ws_server.set_state("listening")
"""

import asyncio
import functools
import http.server
import json
import logging
import threading
from pathlib import Path
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

PORT      = 8765
HTTP_PORT = 3000

_DIST_DIR = Path(__file__).parent / "frontend" / "dist"

_clients: Set[WebSocketServerProtocol] = set()
_loop: asyncio.AbstractEventLoop | None = None
_current_state: str = "idle"
_muted: bool = False
_state_lock = threading.Lock()

logger = logging.getLogger(__name__)


# ── Internal async helpers ────────────────────────────────────────────────────

async def _handler(ws: WebSocketServerProtocol) -> None:
    _clients.add(ws)
    try:
        # Send current state immediately so the UI is in sync on connect
        await ws.send(json.dumps({"state": _current_state, "muted": _muted}))
        # Keep connection alive; we don't expect messages from the browser
        await ws.wait_closed()
    finally:
        _clients.discard(ws)


async def _broadcast(state: str, muted: bool) -> None:
    if not _clients:
        return
    message = json.dumps({"state": state, "muted": muted})
    await asyncio.gather(
        *[ws.send(message) for ws in list(_clients)],
        return_exceptions=True,
    )


async def _serve() -> None:
    async with websockets.serve(_handler, "localhost", PORT):
        await asyncio.Future()  # run forever


# ── Public API ────────────────────────────────────────────────────────────────

def set_state(state: str) -> None:
    """Broadcast a new state to all connected browser clients (thread-safe)."""
    global _current_state
    with _state_lock:
        _current_state = state
        muted = _muted
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(state, muted), _loop)


def send_event(event: dict) -> None:
    """Broadcast an arbitrary JSON event to all connected clients (thread-safe)."""
    if _loop is None:
        return

    async def _do() -> None:
        if not _clients:
            return
        msg = json.dumps(event)
        await asyncio.gather(
            *[ws.send(msg) for ws in list(_clients)],
            return_exceptions=True,
        )

    asyncio.run_coroutine_threadsafe(_do(), _loop)


def set_muted(muted: bool) -> None:
    """Update mute state and broadcast it to connected browser clients."""
    global _muted
    with _state_lock:
        _muted = muted
        state = _current_state
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(state, muted), _loop)


def is_muted() -> bool:
    with _state_lock:
        return _muted


def get_status() -> dict[str, str | bool]:
    with _state_lock:
        return {"state": _current_state, "muted": _muted}


def _serve_http() -> None:
    """Serve frontend/dist/ as static files on HTTP_PORT (no internet needed)."""
    if not _DIST_DIR.exists():
        logger.warning(
            "[http-server] %s not found – run `npm run build` inside frontend/",
            _DIST_DIR,
        )
        return

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(_DIST_DIR), **kwargs)

        def log_message(self, *_):  # silence request logs
            pass

        def do_GET(self):
            if self.path == "/api/status":
                body = json.dumps(get_status()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/mute":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
                muted = bool(data["muted"])
            except (json.JSONDecodeError, KeyError, TypeError):
                self.send_error(400, "Invalid mute payload")
                return

            set_muted(muted)
            body = json.dumps(get_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            # Allow WebSocket connections from the same origin
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    with http.server.ThreadingHTTPServer(("", HTTP_PORT), _QuietHandler) as httpd:
        logger.info("[http-server] serving %s on http://localhost:%d", _DIST_DIR, HTTP_PORT)
        httpd.serve_forever()


def start() -> None:
    """Start both the WebSocket server and the static HTTP server."""
    global _loop

    def _run_ws() -> None:
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_serve())
        except Exception as exc:
            logger.warning("[ws_server] stopped: %s", exc)

    threading.Thread(target=_run_ws,    daemon=True, name="ws-server").start()
    threading.Thread(target=_serve_http, daemon=True, name="http-server").start()

    logger.info("[ws_server] started on ws://localhost:%d", PORT)
