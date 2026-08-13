#!/usr/bin/env bash
#
# Cloud Agent bootstrap for the mtdata development environment.
#
# Idempotent: safe to run repeatedly and against a cached/partially-prepared VM.
# Sets up the Python 3.14 backend (CLI, MCP, Web API) plus the React/Vite web UI.
#
# Platform note: MetaTrader5 is a Windows-only wheel and the sole hard
# dependency that cannot install on Linux. The test suite mocks it (see
# tests/conftest.py) and mtdata imports it lazily, so every dependency EXCEPT
# MetaTrader5 is installed here. Live MT5 connectivity requires Windows.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n\033[1;36m[cloud-agent-install]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. System toolchain + TA-Lib C library (python "TA-Lib" wheel links to it)
# ---------------------------------------------------------------------------
if ! dpkg -s build-essential >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  log "Installing system build tools"
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential git curl pkg-config
fi

if ! ls /usr/lib/libta-lib.so /usr/lib/x86_64-linux-gnu/libta-lib.so >/dev/null 2>&1; then
  log "Installing TA-Lib C library 0.6.4"
  TA_DEB="$(mktemp --suffix=.deb)"
  curl -fLsS \
    https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb \
    -o "$TA_DEB"
  sudo dpkg -i "$TA_DEB"
  sudo ldconfig
  rm -f "$TA_DEB"
fi

# ---------------------------------------------------------------------------
# 2. uv + Python 3.14 (pyproject requires >=3.14)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

log "Ensuring CPython 3.14 is available"
uv python install 3.14

# ---------------------------------------------------------------------------
# 3. Backend virtualenv + dependencies (everything except Windows-only MT5)
# ---------------------------------------------------------------------------
# /usr/bin/c++ is aliased to clang here, which cannot find libstdc++ headers;
# force GNU so source builds (hnswlib, hmmlearn) compile cleanly.
export CC="${CC:-gcc}" CXX="${CXX:-g++}"

if [ ! -x .venv/bin/python ]; then
  log "Creating Python 3.14 virtualenv at .venv"
  uv venv --python 3.14 .venv
fi

log "Resolving backend dependencies (matching CI extras, minus MetaTrader5)"
DEPS_FILE="$(mktemp)"
.venv/bin/python - "$DEPS_FILE" <<'PY'
import sys, tomllib

data = tomllib.load(open("pyproject.toml", "rb"))
project = data["project"]
# Mirror the CI backend install: `.[web,pattern-search-hnsw]`.
selected = (
    project["dependencies"]
    + project["optional-dependencies"]["web"]
    + project["optional-dependencies"]["pattern-search-hnsw"]
)
kept = [
    req for req in selected
    if not req.lower().replace("-", "").startswith("metatrader5")
]
with open(sys.argv[1], "w") as fh:
    fh.write("\n".join(kept) + "\n")
PY

log "Installing backend dependencies + test tools"
uv pip install --python .venv/bin/python -r "$DEPS_FILE" pytest ruff
rm -f "$DEPS_FILE"

log "Installing mtdata (editable, no-deps)"
uv pip install --python .venv/bin/python -e . --no-deps

# ---------------------------------------------------------------------------
# 4. Frontend dependencies + production build (served by mtdata-webapi at /app)
# ---------------------------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  log "Installing web UI dependencies and building the SPA"
  ( cd webui && npm ci && npm run build )
else
  log "npm not found; skipping web UI build"
fi

log "Done. Activate with: source .venv/bin/activate"
