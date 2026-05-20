from __future__ import annotations

import argparse
import os

from openai import OpenAI


def message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    extra = getattr(message, "model_extra", None) or {}
    for key in ("content", "reasoning_content"):
        value = extra.get(key)
        if isinstance(value, str) and value:
            return value
    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        for key in ("content", "reasoning_content"):
            value = dumped.get(key)
            if isinstance(value, str) and value:
                return value
    raise RuntimeError(f"Chat response had no text content: {message!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a vLLM/OpenAI-compatible chat server.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_CHAT_MODEL", "Qwen/Qwen3.5-4B"))
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=args.base_url)
    resp = client.chat.completions.create(
        model=args.model,
        temperature=0.0,
        max_tokens=96,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": "You are an expert SPARQL engineer. Return only SPARQL."},
            {
                "role": "user",
                "content": "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\nQuestion: List five people with names.",
            },
        ],
    )
    print(message_text(resp.choices[0].message).strip())


if __name__ == "__main__":
    main()
