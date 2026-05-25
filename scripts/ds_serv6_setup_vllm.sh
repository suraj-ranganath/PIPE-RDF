#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-pipe-rdf-arr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [ -f "$HOME/miniforge3/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/bin/activate"
else
  echo "Miniforge not found at $HOME/miniforge3. Install Miniforge first." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.util
for name in ["vllm", "openai", "sentence_transformers", "faiss", "SPARQLWrapper"]:
    print(name, "OK" if importlib.util.find_spec(name) else "MISSING")
PY

echo "Environment ready: conda activate $ENV_NAME"
