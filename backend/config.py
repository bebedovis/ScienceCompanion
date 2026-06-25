from pathlib import Path 
from typing import Literal
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings): 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    llm_provider: Literal["openai", "ollama"] = "ollama"
    openai_api_key: str= os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str= "llama3.1:8b"
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cuda"

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "paper_chunks"
    opensearch_username: str = ""
    opensearch_password: str = ""
    opensearch_use_ssl: bool = False
    embedding_dim: int = 1024

    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 50
    default_top_k: int = 6
    rerank_top_n: int = 20
    chunk_size: int = 512
    chunk_overlap: int = 64

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

@lru_cache
def get_settings() -> Settings:
    return Settings()
