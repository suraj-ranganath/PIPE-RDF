from __future__ import annotations

from dataclasses import dataclass
from typing import List
import requests

from openai import OpenAI


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    embed_model: str
    base_url: str


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.provider = config.provider
        self.model = config.model
        self.embed_model = config.embed_model
        self.base_url = config.base_url
        self._openai = None
        if self.provider == "openai":
            self._openai = OpenAI(api_key=config.api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            resp = self._openai.embeddings.create(model=self.embed_model, input=texts)
            return [item.embedding for item in resp.data]
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
        if self.provider == "openai":
            resp = self._openai.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

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
