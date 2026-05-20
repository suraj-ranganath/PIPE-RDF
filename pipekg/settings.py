import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_provider: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    openai_api_key: str
    openai_base_url: str
    openai_chat_model: str
    openai_embed_model: str
    embed_provider: str
    local_embed_model: str
    local_embed_device: str
    sparql_endpoint_url: str


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "gemma3:27b"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        embed_provider=os.getenv("EMBED_PROVIDER", ""),
        local_embed_model=os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3"),
        local_embed_device=os.getenv("LOCAL_EMBED_DEVICE", ""),
        sparql_endpoint_url=os.getenv("SPARQL_ENDPOINT_URL", ""),
    )
