import argparse
import requests
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.settings import get_settings
from pipekg.sparql_client import SparqlClient


def load_run_config(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Run config must be a YAML mapping")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="Path to YAML config (e.g., configs/smoke_test.yaml)")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_run_config(args.config)
    if cfg.get("models", {}).get("chat"):
        settings.ollama_chat_model = cfg["models"]["chat"]
    if cfg.get("models", {}).get("embed"):
        settings.ollama_embed_model = cfg["models"]["embed"]

    # Ollama check
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        names = {t.get("name") for t in tags}
        print("Ollama models:", names)
        missing = []
        if settings.ollama_chat_model not in names:
            missing.append(settings.ollama_chat_model)
        if settings.ollama_embed_model not in names:
            missing.append(settings.ollama_embed_model)
        if missing:
            print("Missing models:", missing)
            raise SystemExit(1)
    except Exception as exc:
        raise SystemExit(f"Ollama check failed: {exc}")

    # SPARQL endpoint check
    if not settings.sparql_endpoint_url:
        raise SystemExit("SPARQL_ENDPOINT_URL not set")
    client = SparqlClient(settings.sparql_endpoint_url)
    res = client.query("SELECT (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }")
    print("Triple count:", res.rows[0].get("triples"))

    print("Preflight OK")


if __name__ == "__main__":
    main()
