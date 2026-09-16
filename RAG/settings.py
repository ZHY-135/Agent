# -*- coding: utf-8 -*-
"""运行配置。仅从环境变量读取，不把密钥写入仓库。"""
import os
from dataclasses import dataclass

from .config import EMBED_DIM


@dataclass(frozen=True)
class Settings:
    backend: str = "memory"
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = EMBED_DIM
    database_url: str | None = None
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        backend = os.getenv("RAG_BACKEND", "memory").lower()
        provider = os.getenv("RAG_EMBEDDING_PROVIDER", "hash").lower()
        dim = int(os.getenv("RAG_EMBED_DIM", str(EMBED_DIM)))
        if backend not in {"memory", "pgvector"}:
            raise ValueError("RAG_BACKEND 只能是 memory 或 pgvector")
        if provider not in {"hash", "openai"}:
            raise ValueError("RAG_EMBEDDING_PROVIDER 只能是 hash 或 openai")
        if dim <= 0:
            raise ValueError("RAG_EMBED_DIM 必须为正数")
        return cls(
            backend=backend,
            embedding_provider=provider,
            embedding_model=os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small"),
            embedding_dim=dim,
            database_url=os.getenv("DATABASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
