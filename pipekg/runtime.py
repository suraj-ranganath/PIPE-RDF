from __future__ import annotations

from typing import Any

from .llm import LLMClient, LLMConfig
from .settings import Settings


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible", "vllm"}


def apply_run_config(settings: Settings, cfg: dict[str, Any]) -> Settings:
    """Apply YAML run config overrides to environment-derived settings."""
    if cfg.get("llm_provider"):
        settings.llm_provider = str(cfg["llm_provider"])
    if cfg.get("ollama_base_url"):
        settings.ollama_base_url = str(cfg["ollama_base_url"])
    if cfg.get("openai_base_url"):
        settings.openai_base_url = str(cfg["openai_base_url"])
    if cfg.get("sparql_endpoint_url"):
        settings.sparql_endpoint_url = str(cfg["sparql_endpoint_url"])

    models = cfg.get("models", {}) or {}
    if models.get("embed_provider"):
        settings.embed_provider = str(models["embed_provider"])
    if models.get("local_embed_device"):
        settings.local_embed_device = str(models["local_embed_device"])

    if models.get("chat"):
        chat_model = str(models["chat"])
        if settings.llm_provider == "ollama":
            settings.ollama_chat_model = chat_model
        else:
            settings.openai_chat_model = chat_model

    if models.get("embed"):
        embed_model = str(models["embed"])
        if (settings.embed_provider or "").lower() in {"sentence_transformers", "local", "local_sentence_transformers"}:
            settings.local_embed_model = embed_model
        elif settings.llm_provider == "ollama":
            settings.ollama_embed_model = embed_model
        else:
            settings.openai_embed_model = embed_model

    return settings


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "ollama":
        return LLMClient(
            LLMConfig(
                provider="ollama",
                api_key="",
                model=settings.ollama_chat_model,
                embed_model=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
                embed_provider=settings.embed_provider,
                local_embed_model=settings.local_embed_model,
                local_embed_device=settings.local_embed_device,
            )
        )
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return LLMClient(
            LLMConfig(
                provider=provider,
                api_key=settings.openai_api_key or "EMPTY",
                model=settings.openai_chat_model,
                embed_model=settings.openai_embed_model,
                base_url=settings.openai_base_url,
                embed_provider=settings.embed_provider,
                local_embed_model=settings.local_embed_model,
                local_embed_device=settings.local_embed_device,
            )
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
