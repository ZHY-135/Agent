# -*- coding: utf-8 -*-
"""运行配置。仅从环境变量读取，不把密钥写入仓库。"""
import os
from dataclasses import dataclass
from typing import Optional

from .config import DEFAULT_LOCAL_MODEL, EMBED_DIM, LOCAL_EMBED_MODELS

# 支持的嵌入器提供方。集中一处定义：避免"校验用一份列表、错误信息里再列一份"，
# 两处迟早会不一致（多了一个提供方却忘了更新提示，用户会以为它不存在）。
EMBEDDING_PROVIDERS = ("hash", "openai", "local")


def resolve_embedding_dim(provider: str, model: str,
                          explicit: Optional[str] = None) -> int:
    """决定向量维度。优先级：显式 `RAG_EMBED_DIM` > 模型维度表 > **报错（不猜）**。

    ★ 为什么不能一律沿用全局默认 `EMBED_DIM=1536`：
      那个值是给 `text-embedding-3-small` 的。换成本地模型后维度不同
      （如 bge-small-zh 是 512 维），而维度会写进 pgvector 的 DDL（`vector(N)`）
      与每一行向量。对不上时错误**延迟到写库那一刻**才爆，报的还是 SQL 层类型错误——
      几乎不可能从那条报错反推出"其实是换了模型"。
      宁可在配置阶段就明确失败，把问题挡在"还没开始建索引"之前。
    """
    if explicit:
        return int(explicit)              # 显式设置永远最高优先：用户知道自己在做什么
    if provider == "local":
        known = LOCAL_EMBED_MODELS.get(model)
        if known is None:
            raise ValueError(
                f"未知的本地模型 {model!r}：无法确定它的向量维度。\n"
                f"    请显式设置 RAG_EMBED_DIM（已知模型："
                f"{'、'.join(sorted(LOCAL_EMBED_MODELS))}）")
        return known
    return EMBED_DIM


@dataclass(frozen=True)
class Settings:
    backend: str = "memory"
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = EMBED_DIM
    database_url: str | None = None
    openai_api_key: str | None = None
    # 本机模型名（仅 `embedding_provider == "local"` 时使用）。
    # 与 `embedding_model` 分开两个字段而不是复用同一个：两者的取值空间完全不同
    # （一个是 OpenAI 的模型名，一个是 HuggingFace 风格的仓库名），
    # 混用一个字段最容易出现的错误是"切到本地后忘了改模型名，于是拿到一个不存在的模型"。
    local_model: str = DEFAULT_LOCAL_MODEL
    # 本机模型的权重缓存目录（仅 local 模式）。
    # ★ 为什么需要它：fastembed 默认把权重下到**用户目录**（`~/.cache/fastembed`
    #   或 Windows 的 AppData）。两个真实场景会被它卡住：
    #     · 受限/沙箱环境不允许写工作区之外 → 下载直接失败；
    #     · 系统盘空间紧张的用户，希望把几百 MB 的模型放到数据盘。
    #   留空则沿用 fastembed 的默认位置（对普通用户最省事）。
    local_cache_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        backend = os.getenv("RAG_BACKEND", "memory").lower()
        provider = os.getenv("RAG_EMBEDDING_PROVIDER", "hash").lower()
        local_model = os.getenv("RAG_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
        dim = resolve_embedding_dim(provider, local_model, os.getenv("RAG_EMBED_DIM"))
        if backend not in {"memory", "pgvector"}:
            raise ValueError("RAG_BACKEND 只能是 memory 或 pgvector")
        if provider not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"RAG_EMBEDDING_PROVIDER 只能是 {' / '.join(EMBEDDING_PROVIDERS)}")
        if dim <= 0:
            raise ValueError("RAG_EMBED_DIM 必须为正数")
        return cls(
            backend=backend,
            embedding_provider=provider,
            embedding_model=os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small"),
            embedding_dim=dim,
            database_url=os.getenv("DATABASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            local_model=local_model,
            local_cache_dir=os.getenv("RAG_LOCAL_CACHE_DIR"),
        )
