from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os
import requests

from openai import OpenAI


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    embed_model: str
    base_url: str
    embed_provider: str = ""
    local_embed_model: str = ""
    local_embed_device: str = ""


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.provider = config.provider
        self.model = config.model
        self.embed_model = config.embed_model
        self.base_url = config.base_url
        self.embed_provider = (config.embed_provider or "").lower()
        self.local_embed_model = config.local_embed_model or config.embed_model
        self.local_embed_device = config.local_embed_device
        self._openai = None
        self._sentence_transformer = None
        if self.provider in {"openai", "openai_compatible", "vllm"}:
            kwargs = {"api_key": config.api_key or "EMPTY"}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self._openai = OpenAI(**kwargs)

    def _maybe_disable_qwen_thinking(self, user: str) -> str:
        disabled = os.getenv("QWEN_NO_THINK", "1").lower() not in {"0", "false", "no"}
        if not disabled or "qwen" not in self.model.lower():
            return user
        if "/no_think" in user.lower():
            return user
        return f"{user.rstrip()}\n\n/no_think"

    @staticmethod
    def _message_text(message: object) -> str:
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
        raise RuntimeError(f"OpenAI-compatible chat response had no text content: {message!r}")

    def _qwen_chat_template_body(self) -> dict | None:
        disabled = os.getenv("QWEN_NO_THINK", "1").lower() not in {"0", "false", "no"}
        if disabled and "qwen" in self.model.lower():
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "EMBED_PROVIDER=sentence_transformers requires the sentence-transformers package."
                ) from exc
            kwargs = {}
            if self.local_embed_device:
                kwargs["device"] = self.local_embed_device
            self._sentence_transformer = SentenceTransformer(self.local_embed_model, **kwargs)
        embeddings = self._sentence_transformer.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embed_provider in {"sentence_transformers", "local", "local_sentence_transformers"}:
            return self._embed_with_sentence_transformers(texts)
        if self.embed_provider == "openai" or (not self.embed_provider and self.provider == "openai"):
            resp = self._openai.embeddings.create(model=self.embed_model, input=texts)
            return [item.embedding for item in resp.data]
        if self.provider in {"openai_compatible", "vllm"} and not self.embed_provider:
            raise RuntimeError(
                "OpenAI-compatible/vLLM chat endpoints usually do not serve embeddings. "
                "Set EMBED_PROVIDER=sentence_transformers and LOCAL_EMBED_MODEL=BAAI/bge-m3."
            )
        # Ollama embeddings (one request per text)
        embeddings: List[List[float]] = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=300,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
        json_schema: dict | None = None,
        timeout_sec: int = 600,
        max_tokens: int | None = None,
    ) -> str:
        if self.provider in {"openai", "openai_compatible", "vllm"}:
            kwargs = {
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": self._maybe_disable_qwen_thinking(user)},
                ],
                "timeout": timeout_sec,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            extra_body = self._qwen_chat_template_body()
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            if json_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pipekg_response",
                        "schema": json_schema,
                    },
                }
            elif json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._openai.chat.completions.create(**kwargs)
            return self._message_text(resp.choices[0].message)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
