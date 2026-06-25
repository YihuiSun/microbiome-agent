#!/usr/bin/env bash
# ============================================================================
# bootstrap.sh — one-time environment setup for microbiome-agent
#
# Tailored to a machine where conda's base Python is too old (3.8). This script
# uses a dedicated conda env (py312) purely as a SOURCE of Python 3.12, then
# builds an independent venv from it and installs everything into that venv.
# Conda just supplies the interpreter; the venv is your real working env.
#
# Run from inside the microbiome-agent folder, with conda available:
#     bash bootstrap.sh
#
# Safe to re-run: it rebuilds the (disposable) venv and reinstalls, but won't
# disturb the py312 conda env if it already exists.
# ============================================================================

set -euo pipefail   # stop on first error; treat unset vars as errors

PY_ENV="py312"          # name of the conda env that provides Python 3.12
PY_VERSION="3.12"       # the Python version we want
VENV_DIR=".venv"        # the project's virtual environment

echo ">> [1/6] Checking that conda is available..."
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found on PATH. Activate your conda (e.g. 'conda activate base') and retry." >&2
  exit 1
fi
conda --version

echo ">> [2/6] Ensuring the '${PY_ENV}' conda env (Python ${PY_VERSION}) exists..."
# `conda env list` prints one env per line; match the name as a whole word.
if conda env list | awk '{print $1}' | grep -qx "${PY_ENV}"; then
  echo "    Found existing '${PY_ENV}' — reusing it."
else
  echo "    Creating '${PY_ENV}' with Python ${PY_VERSION} (this can take a minute)..."
  conda create -n "${PY_ENV}" "python=${PY_VERSION}" -y
fi

echo ">> [3/6] Building a fresh venv from ${PY_ENV}'s Python..."
# Remove any old venv so we always build cleanly on the right interpreter.
rm -rf "${VENV_DIR}"
# `conda run -n ENV ...` runs a command inside that env WITHOUT needing
# `conda activate` (which is unreliable inside scripts). This is the key trick.
conda run -n "${PY_ENV}" python -m venv "${VENV_DIR}"

# From here on we call the venv's own python/pip BY PATH. That avoids relying on
# `source .venv/bin/activate`, which doesn't persist inside a script anyway.
VENV_PY="${VENV_DIR}/bin/python"

echo ">> [4/6] Verifying the venv is Python ${PY_VERSION}..."
ACTUAL="$(${VENV_PY} --version)"
echo "    venv reports: ${ACTUAL}"
case "${ACTUAL}" in
  *"${PY_VERSION}"*) echo "    OK — correct Python." ;;
  *) echo "ERROR: venv is not Python ${PY_VERSION} (${ACTUAL}). Aborting." >&2; exit 1 ;;
esac

echo ">> [5/6] Installing dependencies and the project into the venv..."
"${VENV_PY}" -m pip install --upgrade pip
"${VENV_PY}" -m pip install -r requirements.txt
"${VENV_PY}" -m pip install -e .

echo ">> [6/6] Running the test suite..."
"${VENV_PY}" -m pytest -q

cat <<'DONE'

============================================================================
 Setup complete.

 IMPORTANT: this script installed everything, but it canNOT activate the venv
 in your current terminal (a script can't change its parent shell). To work on
 the project, activate it yourself each time you open a new terminal:

     source .venv/bin/activate

 Then confirm the right Python is in charge with:

     which python      # should end in .../microbiome-agent/.venv/bin/python

 Daily Git rhythm after making changes:

     git add .
     git commit -m "describe what changed"
     git push
============================================================================
DONE
