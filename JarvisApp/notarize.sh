#!/bin/bash
# Notarize and staple Jarvis.app for outside-App-Store distribution.
#
# Usage:
#   ./notarize.sh <path-to-Jarvis.app> <apple-id> <team-id> <app-specific-password>
#
# <app-specific-password>: generate one at appleid.apple.com → Sign-In and Security
#
# Example:
#   ./notarize.sh build/Jarvis.app felix@example.com 9TGSRQR8N4 abcd-efgh-ijkl-mnop

set -euo pipefail

APP="${1:?Usage: $0 <Jarvis.app> <apple-id> <team-id> <app-specific-password>}"
APPLE_ID="${2:?missing apple-id}"
TEAM_ID="${3:?missing team-id}"
PASSWORD="${4:?missing app-specific-password}"
ZIP="${APP}.zip"

echo "▶  Zipping ${APP}…"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "▶  Submitting to Apple notary service…"
xcrun notarytool submit "$ZIP" \
  --apple-id  "$APPLE_ID"  \
  --team-id   "$TEAM_ID"   \
  --password  "$PASSWORD"  \
  --wait

echo "▶  Stapling ticket to app bundle…"
xcrun stapler staple "$APP"

rm -f "$ZIP"
echo "✅  Done. ${APP} is notarized and ready to distribute."
