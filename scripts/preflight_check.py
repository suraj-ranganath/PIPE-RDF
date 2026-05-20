import argparse
import requests
import yaml
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.settings import get_settings
from pipekg.runtime import apply_run_config, build_llm
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
    apply_run_config(settings, cfg)

    # LLM check
    if settings.llm_provider == "ollama":
        try:
            resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            names = {t.get("name") for t in tags}
            print("Ollama models:", names)
            missing = []
            if settings.ollama_chat_model not in names:
                missing.append(settings.ollama_chat_model)
            if (settings.embed_provider or settings.llm_provider) == "ollama" and settings.ollama_embed_model not in names:
                missing.append(settings.ollama_embed_model)
            if missing:
                print("Missing models:", missing)
                raise SystemExit(1)
        except Exception as exc:
            raise SystemExit(f"Ollama check failed: {exc}")
    else:
        try:
            llm = build_llm(settings)
            text = llm.chat(
                system="Return exactly OK.",
                user="Return exactly OK.",
                temperature=0.0,
                max_tokens=8,
                timeout_sec=30,
            )
            print("Chat check:", text.strip()[:80])
        except Exception as exc:
            raise SystemExit(f"OpenAI-compatible chat check failed: {exc}")

    if (settings.embed_provider or "").lower() in {"sentence_transformers", "local", "local_sentence_transformers"}:
        try:
            llm = build_llm(settings)
            emb = llm.embed_texts(["PIPE-RDF preflight embedding check"])
            print("Embedding dim:", len(emb[0]) if emb else 0)
        except Exception as exc:
            raise SystemExit(f"Local embedding check failed: {exc}")

    # SPARQL endpoint check
    if not settings.sparql_endpoint_url:
        raise SystemExit("SPARQL_ENDPOINT_URL not set")
    client = SparqlClient(settings.sparql_endpoint_url)
    res = client.query("SELECT (COUNT(*) AS ?triples) WHERE { ?s ?p ?o }")
    print("Triple count:", res.rows[0].get("triples"))

    print("Preflight OK")


if __name__ == "__main__":
    main()
