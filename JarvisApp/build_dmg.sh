#!/bin/bash
# Build a distributable DMG for Jarvis.app.
#
# Requires: brew install create-dmg
#
# Usage:
#   ./build_dmg.sh <path-to-Jarvis.app>
#
# Example:
#   ./build_dmg.sh build/Jarvis.app

set -euo pipefail

APP="${1:?Usage: $0 <path-to-Jarvis.app>}"

if ! command -v create-dmg &>/dev/null; then
  echo "❌  create-dmg not found — install it with: brew install create-dmg"
  exit 1
fi

OUT="Jarvis.dmg"
[ -f "$OUT" ] && rm "$OUT"

create-dmg \
  --volname "Jarvis"           \
  --window-size 600 400        \
  --icon-size 128              \
  --icon "Jarvis.app" 175 190  \
  --app-drop-link 425 190      \
  "$OUT"                       \
  "$APP"

echo "✅  ${OUT} created."
