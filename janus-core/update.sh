#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(which python3 || which python || echo "python3")"

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/update_skills.py" "$@"

