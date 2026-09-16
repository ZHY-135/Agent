# -*- coding: utf-8 -*-
"""逐步调试脚本：把一条查询在各环节的中间结果全部打印出来。

    python -m RAG.debug_trace                          # 用项目根 docs/ 与默认问题
    python -m RAG.debug_trace --docs ./corpus --query "错误码 0x80070005"
    python -m RAG.debug_trace --stage chunk            # 只看切分阶段
    python -m RAG.debug_trace --max-len 200            # 控制每段打印长度

为什么需要它：跑 `python -m RAG.main demo` 只能看到**最终结果**（prompt / 答案）。
当结果不对时，你无法知道是哪一环出的问题——是切分把答案切断了？
是向量没召回？还是重排把正确的块挤掉了？
本脚本按流水线顺序逐环打印，让每一环的输入输出都可见。

阶段划分（--stage 可选其中一个或 all）：
    load    文件加载与内容指纹
    chunk   切分：结构块 → 父块 → 滑窗子块（含字符偏移）
    embed   把子块变成向量（看几个维度、取值量级）
    index   建索引 + 统计
    recall  双通道召回：向量通道 / 关键词通道 **分开看**
    fuse    RRF 融合后的排名变化
    agg     按父块聚合（去重）
    rerank  MMR 多样性重排
    pack    上下文预算装填 + 最终 prompt
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from . import config as C
from .chunkers import MarkdownSplitter, build_parent_child
from .embedders import HashEmbedder, embed_all
from .loaders import file_fingerprint, load_text
from .logging_setup import configure_logging
from .pipeline import RAGPipeline
from .rerankers import mmr
from .retrievers import aggregate_by_parent, rrf_fuse
from .stores import MemoryStore

DEFAULT_QUERY = "连接池最大连接数是多少"


# ============================== 打印辅助 ==============================

def section(title: str) -> None:
    print(f"\n{'=' * 78}\n▶ {title}\n{'=' * 78}")


def cut(text: str, limit: int) -> str:
    """截断长文本并显式标注被截掉多少字符——避免"以为看全了"。"""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]}…（另 {len(flat) - limit} 字符）"


def show_rows(rows, key_fn, limit: int = 8) -> None:
    for i, row in enumerate(rows[:limit]):
        print(f"    [{i + 1}] {key_fn(row)}")
    if len(rows) > limit:
        print(f"    …（共 {len(rows)} 条，仅显示前 {limit} 条）")


# ============================== 各阶段 ==============================

def vector_preview(vec, limit: int = 6) -> str:
    """显示向量的**非零**维度。

    ⚠ 不能只打印前 6 维：HashEmbedder 的向量是"按哈希投票"得到的，
    1536 维里通常只有几十个非零值，前几维几乎必然是 0.000——
    照原样打印会让人误以为"向量全是零、嵌入没生效"（这正是初版的表现）。
    """
    nz = [(i, x) for i, x in enumerate(vec) if x]
    nz.sort(key=lambda t: -t[1])                      # 按权重从大到小
    head = ", ".join(f"#{i}={x:.3f}" for i, x in nz[:limit])
    return f"非零维度 {len(nz)}/{len(vec)}，权重最大的 {min(limit, len(nz))} 个：{head}"


def stage_load(docs: Path, max_len: int, pick: Optional[str] = None) -> dict:
    section("① 加载：文件 → 文本 + 内容指纹")
    files = sorted(p for p in docs.rglob("*") if p.is_file()
                   and p.suffix.lower() in {".md", ".markdown", ".txt", ".html", ".htm"})
    if not files:
        raise SystemExit(f"{docs} 下没有可用的文档")
    target = None
    for path in files:
        fp = file_fingerprint(str(path))
        text = load_text(str(path))
        print(f"  · {path.name}")
        print(f"      指纹 {fp}（内容哈希前 16 位，用于『没变就跳过』）")
        print(f"      字符数 {len(text)}，开头 {text[:60]!r}…")
        # 不指定 --file 时默认挑"最大的那个文件"：
        # 小文件切不出几个块，看不出滑窗重叠、聚合去重这些现象
        if pick:
            if path.name == pick:
                target = (path, text, fp)
        elif target is None or len(text) > target[1].__len__():
            target = (path, text, fp)
    if target is None:
        raise SystemExit(f"没有找到名为 {pick!r} 的文件（可选："
                         f"{', '.join(p.name for p in files)}）")
    print(f"\n  本脚本用 {target[0].name}（{len(target[1])} 字符）继续演示")
    return {"path": target[0], "text": target[1], "fingerprint": target[2]}


def stage_chunk(loaded: dict, max_len: int) -> dict:
    text, source = loaded["text"], str(loaded["path"])

    section("② 切分第 1 步：整篇文档 → 结构块（按空行/标题/代码围栏切）")
    blocks = MarkdownSplitter.split(text)
    print(f"  共切出 {len(blocks)} 个结构块")
    show_rows(blocks, lambda b: f"标题路径={b[0]!r}  文本={cut(b[1], max_len)!r}", limit=6)
    print("  说明：标题路径（面包屑）会作为前缀拼到父块上，让每个块都知道自己属于哪一节")

    section("② 切分第 2 步：结构块 → 父块 + 子块（1:N）")
    idx = build_parent_child(text, source)
    parents, children = idx["parents"], idx["children"]
    print(f"  父块 {len(parents)} 个（上限 {C.MAX_PARENT_CHARS} 字符），"
          f"子块 {len(children)} 个（上限 {C.MAX_CHILD_CHARS} 字符，"
          f"重叠 {C.CHILD_OVERLAP}）")

    first_parent = list(parents.values())[0]
    print("\n  ── 第 1 个父块 ──")
    print(f"      parent_id = {first_parent['parent_id']}")
    print(f"      heading   = {first_parent['heading']!r}")
    print(f"      char_len  = {first_parent['char_len']}")
    print(f"      text      = {cut(first_parent['parent_text'], max_len)!r}")

    kids = [c for c in children if c["parent_id"] == first_parent["parent_id"]]
    print(f"\n  ── 它的 {len(kids)} 个子块（注意 char_start/char_end 与重叠）──")
    for c in kids:
        print(f"      #{c['child_index']} 位置[{c['char_start']}:{c['char_end']}] "
              f"长度{len(c['child_text'])}  {cut(c['child_text'], max_len)!r}")
    if len(kids) > 1:
        a, b = kids[0], kids[1]
        overlap_chars = a["char_end"] - b["char_start"]
        print(f"      重叠验证：子块0 结束于 {a['char_end']}，子块1 开始于 {b['char_start']}"
              f" → 重叠 {overlap_chars} 字符")
    else:
        print("      （该父块太短，只切出 1 个子块，看不出重叠）")

    print("\n  说明：子块文本是父块的**逐字切片**，所以 char_start/char_end 可信、"
          "引用能溯源")
    return {"parents": parents, "children": children, "text": text,
            "path": loaded["path"]}


def stage_embed(chunked: dict, max_len: int) -> object:
    section("③ 嵌入：子块文本 → 向量")
    embedder = HashEmbedder()
    sample = chunked["children"][:3]
    vecs = embed_all(embedder, [c["child_text"] for c in sample])
    print(f"  用 HashEmbedder（伪向量，维度 {embedder.dim}）")
    for c, v in zip(sample, vecs):
        print(f"  · {cut(c['child_text'], 40)!r}")
        print(f"      维度 {len(v)}：{vector_preview(v)}")
        print(f"      模长 {sum(x * x for x in v) ** 0.5:.4f}（L2 归一化后应为 1.0000）")
    print("\n  说明：向量数量必须与子块数量一致且顺序对应，否则向量会挂到错误的内容上")
    return embedder


def stage_index(embedder, chunked: dict) -> RAGPipeline:
    section("④ 建索引：写入存储 + 重建 BM25")
    pipeline = RAGPipeline(embedder, MemoryStore(), use_bm25=True, use_mmr=True)
    text, source = chunked["text"], str(chunked["path"])
    idx = build_parent_child(text, source)
    parents, children = idx["parents"], idx["children"]
    vecs = embed_all(embedder, [c["child_text"] for c in children])
    for c, v in zip(children, vecs):
        c["embedding"] = v
        c["source"] = source
    pipeline.store.replace_document(source, file_fingerprint(source),
                                   list(parents.values()), children)
    pipeline._rebuild_bm25()
    print(f"  已写入：父块 {len(parents)}，子块 {len(children)}")
    n_docs = len(pipeline.store.list_sources("."))   # 走接口，不读实现内部属性
    print(f"  store 已索引文档数 = {n_docs}（指纹记录，用于增量跳过）")
    print(f"  BM25 索引 = {'已建立' if pipeline._bm25 else '无'}，"
          f"词表大小 {len(pipeline._bm25.df) if pipeline._bm25 else 0}")
    return pipeline


def stage_recall(pipeline: RAGPipeline, query: str, top_k: int, max_len: int):
    section(f"⑤ 双通道召回：query = {query!r}")
    qv = embed_all(pipeline.embedder, [query])[0]
    cand = top_k * C.CANDIDATE_MUL
    print(f"  候选数 = top_k({top_k}) × CANDIDATE_MUL({C.CANDIDATE_MUL}) = {cand}")
    print(f"  查询向量：{vector_preview(qv)}")

    print("\n  ── 通道 ①：向量检索（余弦相似度）──")
    print("     本调试脚本传 min_sim=0.0（关闭阈值过滤），这样即使分数很低的所有候选"
          "都能看到；")
    print(f"     生产默认阈值是 config.MIN_SIM={C.MIN_SIM}，低于它的候选会被直接丢掉")
    vec_hits = pipeline.store.search_children(qv, top_k=cand, min_sim=0.0)
    show_rows(vec_hits, lambda h: f"score={h['score']:.4f}  {cut(h['child_text'], max_len)!r}")

    print("\n  ── 通道 ②：BM25 关键词检索（只看字面 token 是否命中）──")
    kw_hits = pipeline._bm25.search(query, top_k=cand) if pipeline._bm25 else []
    show_rows(kw_hits, lambda h: f"score={h['score']:.4f}  {cut(h['child_text'], max_len)!r}")
    if not kw_hits:
        print("     （无命中：查询里的字面 token 没在任何子块中出现）")
    print("\n  说明：向量通道分数落在 [0,1]，BM25 分数无上界——**两者不可直接相加**，"
          "这正是下一步要用 RRF 的原因")
    return qv, vec_hits, kw_hits


def stage_fuse(vec_hits, kw_hits, top_k: int, max_len: int):
    section("⑥ RRF 融合：只看排名，不看分数")
    print(f"  公式：融合分 = Σ 权重 × 1/(k + 排名)，k = {C.RRF_K}")
    print(f"  即排名第 1 得 {1 / (C.RRF_K + 1):.4f}，第 2 得 {1 / (C.RRF_K + 2):.4f}…")

    fused = rrf_fuse([vec_hits, kw_hits])
    print(f"\n  融合后 {len(fused)} 条（同一 child 被两路命中时分数叠加 → 排名上升）")
    show_rows(fused, lambda h: f"融合分={h['score']:.4f}  {cut(h['child_text'], max_len)!r}")

    if vec_hits and kw_hits:
        vec_top = vec_hits[0]["child_id"]
        rank_in_fused = next((i + 1 for i, h in enumerate(fused)
                             if h["child_id"] == vec_top), None)
        print(f"\n  观察：向量通道的第 1 名，在融合结果里排第 {rank_in_fused}"
              f"（若 >1 说明关键词通道把别的块顶上来了）")
    return fused


def stage_agg(fused, top_k: int, max_len: int, pipeline: RAGPipeline):
    section("⑦ 按父块聚合：同一父块只留最高分的一条")
    # 先展示"聚合前同一父块出现多次"的现象
    counts: dict[str, int] = {}
    for h in fused:
        counts[h["parent_id"]] = counts.get(h["parent_id"], 0) + 1
    dup = {pid: n for pid, n in counts.items() if n > 1}
    print(f"  融合结果涉及 {len(counts)} 个父块；其中 {len(dup)} 个父块被命中多次：")
    for pid, n in list(dup.items())[:3]:
        print(f"      {pid[:8]}… 命中 {n} 次 → 聚合后只保留 1 条")

    agg = aggregate_by_parent(fused)
    print(f"\n  聚合后 {len(agg)} 条（已在去重，尚未截断）")
    show_rows(agg, lambda h: f"score={h['score']:.4f}  parent={h['parent_id'][:8]}…  "
                             f"{cut(h['child_text'], max_len)!r}")

    truncated = agg[:top_k * C.CANDIDATE_MUL]
    print(f"\n  截断到 top_k×{C.CANDIDATE_MUL} = {len(truncated)} 条作为重排候选")
    print("  ★ 顺序不能反：若先截断再聚合，前 3 条可能全属同一父块，去重后只剩 1 条")
    return truncated


def stage_rerank(pipeline: RAGPipeline, agg, top_k: int, max_len: int):
    section("⑧ MMR 多样性重排")
    print(f"  λ = {C.MMR_LAMBDA}（1=只看相关性，0=只看多样性）")
    vectors = pipeline.store.get_child_vectors([h["child_id"] for h in agg])
    reranked = mmr(agg, vectors, top_k=top_k)

    order_before = [h["child_id"][:8] for h in agg[:top_k]]
    order_after = [h["child_id"][:8] for h in reranked]
    print(f"\n  重排前前 {top_k}：   {order_before}")
    print(f"  重排后前 {top_k}：   {order_after}")
    if order_before == order_after:
        print("  （顺序未变：本轮候选之间没有高度重复的内容，多样性项没有发挥作用）")
    show_rows(reranked, lambda h: f"score={h['score']:.4f}  {cut(h['child_text'], max_len)!r}")
    return reranked


def stage_pack(pipeline: RAGPipeline, hits, query: str, max_len: int):
    section("⑨ 上下文预算装填 + 最终 prompt")
    parents = pipeline.store.get_parents([h["parent_id"] for h in hits])
    from .context import pack_context
    packed = pack_context(hits, parents, query)

    print(f"  预算 {packed['budget']} token，固定开销（system 400 + 历史 0 + 问题）"
          f" 已先扣掉")
    print(f"  装入 {len(packed['blocks'])} 块，用 {packed['used_tokens']} token，"
          f"利用率 {packed['utilization'] * 100:.0f}%，丢弃 {len(packed['dropped'])} 块")
    for b in packed["blocks"]:
        print(f"\n      [{b['ref']}] kind={b['kind']} tokens={b['tokens']} "
              f"score={b['score']:.4f}")
        print(f"          来源={Path(b['source']).name}  标题={b['heading']!r}")
        print(f"          文本={cut(b['text'], max_len)!r}")
    if packed["dropped"]:
        print("\n  ⚠ 被预算挤掉的块（这些内容不会进入 prompt）：")
        show_rows(packed["dropped"], lambda h: cut(h["child_text"], max_len), limit=3)

    from .context import render_prompt
    prompt = render_prompt(packed, query)
    print(f"\n  ── 最终 prompt（共 {len(prompt)} 字符）──")
    print("  " + "\n  ".join(prompt.splitlines()))
    return packed


# ============================== 主流程 ==============================

STAGES = ("load", "chunk", "embed", "index", "recall", "fuse", "agg", "rerank", "pack")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="逐步打印各环节中间结果")
    parser.add_argument("--docs", default="docs",
                        help="语料目录（默认项目根下的 docs/；路径相对项目根）")
    parser.add_argument("--file", default=None,
                        help="只用指定文件演示（默认挑最大的那个：小文件切不出几块，"
                             "看不出滑窗/聚合现象）")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=90,
                        help="每段打印的最大字符数（调大看全文）")
    parser.add_argument("--stage", default="all",
                        help=f"只跑某阶段，可选 {', '.join(STAGES)} 或 all")
    parser.add_argument("--verbose", action="store_true",
                        help="额外打开库内部日志（会与本节打印交错，便于对照）")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose, quiet=not args.verbose)
    wanted = set(STAGES) if args.stage == "all" else {args.stage}
    if not wanted <= set(STAGES):
        raise SystemExit(f"--stage 只能是 {', '.join(STAGES)} 或 all")

    print(f"语料目录：{Path(args.docs).resolve()}")
    print(f"查询：{args.query!r}    top_k={args.top_k}    --max-len={args.max_len}")

    loaded = stage_load(Path(args.docs), args.max_len, args.file) if "load" in wanted else None
    if loaded is None:                       # 后续阶段都依赖加载结果
        files = sorted(p for p in Path(args.docs).rglob("*") if p.is_file())
        pick = next((p for p in files if p.name == args.file), files[0])
        loaded = {"path": pick, "text": load_text(str(pick)),
                  "fingerprint": file_fingerprint(str(pick))}

    chunked = stage_chunk(loaded, args.max_len) if "chunk" in wanted else None
    if chunked is None:
        idx = build_parent_child(loaded["text"], str(loaded["path"]))
        chunked = {"parents": idx["parents"], "children": idx["children"],
                   "text": loaded["text"], "path": loaded["path"]}

    embedder = stage_embed(chunked, args.max_len) if "embed" in wanted else HashEmbedder()
    pipeline = stage_index(embedder, chunked)

    qv, vec_hits, kw_hits = stage_recall(pipeline, args.query, args.top_k, args.max_len)
    fused = stage_fuse(vec_hits, kw_hits, args.top_k, args.max_len)
    agg = stage_agg(fused, args.top_k, args.max_len, pipeline)
    hits = stage_rerank(pipeline, agg, args.top_k, args.max_len) if "rerank" in wanted else agg[:args.top_k]
    if "pack" in wanted:
        stage_pack(pipeline, hits, args.query, args.max_len)

    section("完成")
    print("  想改哪个参数看效果？直接编辑 RAG/config.py（所有可调项都在那一个文件里）：")
    print(f"    MAX_PARENT_CHARS={C.MAX_PARENT_CHARS}  MAX_CHILD_CHARS={C.MAX_CHILD_CHARS}  "
          f"CHILD_OVERLAP={C.CHILD_OVERLAP}")
    print(f"    TOP_K={C.TOP_K}  CANDIDATE_MUL={C.CANDIDATE_MUL}  MIN_SIM={C.MIN_SIM}  "
          f"RRF_K={C.RRF_K}")
    print(f"    MMR_LAMBDA={C.MMR_LAMBDA}  CONTEXT_BUDGET={C.CONTEXT_BUDGET}")
    print("\n  想用真正的断点调试：")
    print("    python -m pdb -m RAG.debug_trace          # 逐行执行，n 下一步 / s 进入 / "
          "p 变量 / c 继续")


if __name__ == "__main__":
    main()
