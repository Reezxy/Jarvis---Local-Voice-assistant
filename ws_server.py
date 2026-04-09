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
import os
import threading
from pathlib import Path
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

PORT      = 8765
HTTP_PORT = 3000

# When running inside the macOS app, JARVIS_DIST_DIR points to the pre-built
# frontend bundled in Contents/Resources/dist/.
# When running locally (start.command), fall back to frontend/dist/ next to this file.
_dist_env = os.environ.get("JARVIS_DIST_DIR")
_DIST_DIR = (
    Path(_dist_env)
    if _dist_env and Path(_dist_env).is_dir()
    else Path(__file__).parent / "frontend" / "dist"
)

_clients: Set[WebSocketServerProtocol] = set()
_loop: asyncio.AbstractEventLoop | None = None
_current_state: str = "idle"
_muted: bool = False
_state_lock = threading.Lock()
_start_lock = threading.Lock()
_servers_started = False

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
    async with websockets.serve(_handler, "127.0.0.1", PORT):
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


def _cors_end_headers(handler: http.server.BaseHTTPRequestHandler) -> None:
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def _handle_api_status(handler: http.server.BaseHTTPRequestHandler) -> None:
    body = json.dumps(get_status()).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    _cors_end_headers(handler)
    handler.wfile.write(body)


def _handle_api_mute(handler: http.server.BaseHTTPRequestHandler) -> None:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        data = json.loads(raw.decode("utf-8"))
        muted = bool(data["muted"])
    except (json.JSONDecodeError, KeyError, TypeError):
        handler.send_error(400, "Invalid mute payload")
        return

    set_muted(muted)
    body = json.dumps(get_status()).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    _cors_end_headers(handler)
    handler.wfile.write(body)


class _APIOnlyHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP handler if frontend/dist is missing — still binds :3000 for the macOS app."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/status":
            _handle_api_status(self)
            return
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            html = (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/><title>Jarvis</title></head>"
                "<body style=\"font-family:system-ui;padding:2rem\">"
                "<h1>Jarvis</h1>"
                "<p>Die Web-UI fehlt. Im Projektordner ausführen:</p>"
                "<pre style=\"background:#eee;padding:1rem\">cd frontend && npm install && npm run build</pre>"
                "<p>Dann Jarvis neu starten.</p></body></html>"
            )
            b = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            _cors_end_headers(self)
            self.wfile.write(b)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/api/mute":
            _handle_api_mute(self)
            return
        self.send_error(404)


def _serve_http() -> None:
    """Serve frontend/dist/ on HTTP_PORT; if dist is missing, still listen (API + stub page)."""
    has_dist = _DIST_DIR.is_dir()

    if not has_dist:
        msg = (
            f"[http-server] {_DIST_DIR} fehlt — starte nur API auf Port {HTTP_PORT} "
            "(cd frontend && npm run build für die volle UI)\n"
        )
        logger.warning(msg.strip())
        print(msg, flush=True)

        with http.server.ThreadingHTTPServer(("", HTTP_PORT), _APIOnlyHandler) as httpd:
            logger.info("[http-server] API-only http://localhost:%d", HTTP_PORT)
            print(f"[http-server] listening on http://127.0.0.1:{HTTP_PORT} (API-only)\n", flush=True)
            httpd.serve_forever()
        return

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(_DIST_DIR), **kwargs)

        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/api/status":
                _handle_api_status(self)
                return
            super().do_GET()

        def do_POST(self):
            if self.path.split("?", 1)[0] == "/api/mute":
                _handle_api_mute(self)
                return
            self.send_error(404)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    with http.server.ThreadingHTTPServer(("", HTTP_PORT), _QuietHandler) as httpd:
        logger.info("[http-server] serving %s on http://localhost:%d", _DIST_DIR, HTTP_PORT)
        print(f"[http-server] serving { _DIST_DIR } on http://127.0.0.1:{HTTP_PORT}\n", flush=True)
        httpd.serve_forever()


def start() -> None:
    """Start both the WebSocket server and the static HTTP server."""
    global _loop, _servers_started

    with _start_lock:
        if _servers_started:
            return
        _servers_started = True

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
