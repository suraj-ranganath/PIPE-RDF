import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipekg.settings import get_settings
from pipekg.llm import LLMClient, LLMConfig
from pipekg.sparql_client import SparqlClient
from pipekg.evaluation import parse_valid_sparql
from pipekg.vector_store import FaissStore
import numpy as np


def extract_sparql(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
        else:
            text = parts[-1]
    text = text.replace("sparql", "").strip()
    if text.lower().startswith("select") or text.lower().startswith("ask"):
        return text

    # Fallback: find first SELECT/ASK
    for token in ("SELECT", "ASK"):
        idx = text.upper().find(token)
        if idx != -1:
            return text[idx:]
    return text

def build_llm(settings):
    if settings.llm_provider == "ollama":
        return LLMClient(
            LLMConfig(
                provider="ollama",
                api_key="",
                model=settings.ollama_chat_model,
                embed_model=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
            )
        )
    return LLMClient(
        LLMConfig(
            provider="openai",
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            embed_model=settings.openai_embed_model,
            base_url="",
        )
    )


def main() -> None:
    settings = get_settings()
    if not settings.sparql_endpoint_url:
        raise SystemExit("SPARQL_ENDPOINT_URL is not set")

    llm = build_llm(settings)
    client = SparqlClient(settings.sparql_endpoint_url)

    prefixes = """
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
""".strip()

    schema = """
Classes:
- foaf:Person
Properties:
- foaf:name (person name)
""".strip()

    tests = [
        {
            "category": "generic",
            "question": "List 5 people and their names.",
            "expected_type": "SELECT",
        },
        {
            "category": "counting",
            "question": "How many people are in the graph?",
            "expected_type": "SELECT",
        },
        {
            "category": "yesno",
            "question": "Is there any person with a name?",
            "expected_type": "ASK",
        },
    ]

    results = []
    for item in tests:
        system = "You are an expert SPARQL engineer. Use only the provided prefixes and schema. Return only SPARQL."
        user = f"{prefixes}\n\nSchema:\n{schema}\n\nQuestion: {item['question']}\nExpected query form: {item['expected_type']}"

        start = time.time()
        sparql_raw = llm.chat(system=system, user=user, temperature=0.2)
        sparql = extract_sparql(sparql_raw)
        if "PREFIX" not in sparql.upper():
            sparql = prefixes + "\n" + sparql
        latency = (time.time() - start) * 1000

        parse_ok = parse_valid_sparql(sparql)
        exec_ok = False
        answers = []
        error = None

        if parse_ok:
            try:
                res = client.query(sparql)
                exec_ok = True
                if res.boolean is not None:
                    answers = [str(res.boolean)]
                else:
                    # flatten first row only for quick test
                    if res.rows:
                        answers = [str(v) for v in res.rows[0].values()]
            except Exception as exc:
                error = str(exc)

        results.append(
            {
                "category": item["category"],
                "question": item["question"],
                "sparql": sparql,
                "parse_valid": parse_ok,
                "exec_success": exec_ok,
                "answers_sample": answers,
                "error": error,
                "latency_ms": latency,
            }
        )

    # Embedding + FAISS quick check
    texts = [r["question"] for r in results]
    emb = np.array(llm.embed_texts(texts), dtype="float32")
    store = FaissStore.build(emb, [{"question": t} for t in texts])
    query_emb = np.array(llm.embed_texts(["people names"]), dtype="float32")
    top = store.search(query_emb, k=2)[0]

    out_dir = Path("artifacts/quick_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quick_test_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "quick_test_retrieval.json").write_text(json.dumps(top, indent=2))

    print("Quick test complete. Results written to artifacts/quick_test/")


if __name__ == "__main__":
    main()
