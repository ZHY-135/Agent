# -*- coding: utf-8 -*-
"""压力验证上下文预算与 top_k 截断；用法：python -m RAG.stress_test。

临时文档默认建在包目录下的 .pytest_tmp/（可用 RAG_TEST_TMPDIR 覆盖），
而不是系统 temp——受限沙箱下系统 temp 不可写，压测必须能在任意环境自证通过。
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from RAG import config as C
from RAG.context import estimate_tokens
from RAG.embedders import HashEmbedder
from RAG.pipeline import RAGPipeline
from RAG.stores import MemoryStore


def make_document() -> str:
    return "\n\n".join(
        f"## 章节 {index}\n\n" + f"这是第 {index} 节关于数据库连接池的详细说明内容。" * 25
        for index in range(40)
    )


def make_workdir() -> Path:
    """优先用包内临时目录，避免系统 temp 不可写导致整个压测跑不起来。

    ⚠ 某些受限沙箱（如 workspace-write）会禁止向「运行期新建的目录」写入文件——
       此时可预设 RAG_TEST_TMPDIR 指向一个已存在的可写目录（如 ./tmp）。
       两者都不可用时才退回系统 temp，并保留原始报错让用户看到真实原因。
    """
    for root in (os.environ.get("RAG_TEST_TMPDIR"),
                 str(Path(__file__).resolve().parent / ".pytest_tmp")):
        if not root:
            continue
        path = Path(root)
        try:
            (path / "work").mkdir(parents=True, exist_ok=True)
            (path / "work" / ".probe").write_text("ok", encoding="utf-8")
            return path / "work"
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="rag-stress-"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    temp_dir = make_workdir()
    try:
        docs_dir = temp_dir
        document = make_document()
        (docs_dir / "big.md").write_text(document, encoding="utf-8")

        pipeline = RAGPipeline(HashEmbedder(), MemoryStore(), use_bm25=True, use_mmr=True)
        stat = pipeline.index(str(docs_dir))
        print(f"索引：父块 {stat['parents']} / 子块 {stat['children']}，文档 {len(document)} 字符")

        no_budget = pipeline.query("连接池", top_k=20, min_sim=0.0)
        parent_tokens = sum(estimate_tokens(parent["parent_text"])
                            for parent in no_budget["parents_used"].values())
        fixed_tokens = 400 + 3000 + estimate_tokens("连接池")
        print(f"无预算场景：{parent_tokens + fixed_tokens} token，预算为 {C.CONTEXT_BUDGET}")

        bounded = pipeline.query("连接池", top_k=20, history_tokens=3000, min_sim=0.0)
        packed = bounded["packed"]
        assert packed["used_tokens"] <= packed["budget"]
        print(f"预算控制：{packed['used_tokens']} / {packed['budget']} token，丢弃 {len(packed['dropped'])} 块")

        without_mmr = RAGPipeline(HashEmbedder(), MemoryStore(), use_bm25=True, use_mmr=False)
        without_mmr.index(str(docs_dir))
        assert len(without_mmr.query("连接池", top_k=5, min_sim=0.0)["hits"]) <= 5
        assert len(pipeline.query("连接池", top_k=5, min_sim=0.0)["hits"]) <= 5
        print("压力验证通过")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
