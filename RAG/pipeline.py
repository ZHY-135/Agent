# -*- coding: utf-8 -*-
"""⑨ 编排层：把各模块串成 index() / retrieve() / query() / answer() 主流程。

只做编排，不含业务逻辑——方便单独替换任意一层。

--------------------------------------------------------------------------------
四个入口的分工（读懂这个文件的关键）
--------------------------------------------------------------------------------
    index()    离线：加载 → 切分 → 嵌入 → 存储 → 重建 BM25
    retrieve() 在线检索：双通道召回 → RRF 融合 → 父块聚合 → MMR → 预算装填
    query()    检索 + 渲染 prompt（不调用 LLM）
    answer()   检索 + 生成 + 引用回填校验 + 拒答（需要注入 Generator）

为什么要有 retrieve() 这一层（而不是把检索写在 query 里）：
因为**评测和线上必须是同一条检索路径**。如果评测调 query()、线上调 answer()，
两边的召回行为一旦有细微差异（比如一边多了重排、一边少了阈值），
评测数字就不再代表线上的真实表现——这是评测体系失效最隐蔽的方式。
所以两者都从 retrieve() 取结果，只有后半段（要不要调 LLM）不同。

--------------------------------------------------------------------------------
完整数据流（建议对着这段读代码）
--------------------------------------------------------------------------------
    索引阶段
      文件 → load_all()              得到 (路径, 文本, 指纹)；指纹相同就整篇跳过
           → build_parent_child()    切出父块 + 子块（1:N）
           → embed_all()             给子块算向量（批量 + 重试 + 维度校验）
           → store.replace_document() 事务内"先删后插"，实现幂等更新

    检索阶段
      问题 → embed_all([query])      把问题也变成向量（必须用同一个 embedder）
           → search_children()        向量通道：余弦近邻召回 top_k×4
           → BM25.search()            关键词通道：精确 token 匹配
           → rrf_fuse()               两路按排名融合
           → aggregate_by_parent()    同一父块只留一条（★ 必须在截断之前）
           → mmr()                    多样性去重
           → get_parents()            回表取父块全文
           → pack_context()           按 token 预算装填 + 编号

    生成阶段（answer 才有）
      → render_prompt(strict)        注入"只依据资料 + 强制标注引用"的约束
      → Generator.answer()           生成前门禁 → 调 LLM → 引用回填校验
"""
import hashlib
import os
import time
from typing import Callable, Dict, Optional

from . import config as C
from .chunkers import build_parent_child, chunk_signature
from .context import pack_context, render_prompt
from .embedders import Embedder, embed_all
from .generators import Generator, support_score
from .loaders import load_converted
from .logging_setup import get_logger
from .rerankers import mmr
from .retrievers import BM25, aggregate_by_parent, rrf_fuse
from .stores import VectorStore

logger = get_logger(__name__)


def _noop_trace(**_kwargs) -> None:
    """默认观测钩子：什么都不做。

    为什么默认给一个空函数而不是在每个调用点判断 `if self._trace`：
    判断散落在十几处，早晚会漏一个；给空函数则调用点统一、
    省掉一次分支，且"没有钩子"这条路径的行为可被完全静态确认。
    """
    return None


def compute_index_signature(embedder: Optional[Embedder] = None) -> str:
    """索引签名：决定"同样内容是否需要重新切分/重算向量"的全部因素。

    = 切分参数签名（chunkers.chunk_signature）+ 嵌入器身份（embedder.identity）

    ★ 为什么必须有它（本文件修掉的一个静默错误）：
      原先的增量判据只有**文件内容哈希**。于是把 `MAX_CHILD_CHARS` 从 300 改到 150、
      或者换掉嵌入模型之后重跑索引，每个文件都被判为"内容没变"而跳过：
      日志里是一片漂亮的 `skipped=N`，而实际跑的仍是**旧参数切出来、旧模型算出来的块**。
      这类"改了没生效、还不报错"的问题，比任何崩溃都难排查。
      把签名并进指纹后：签名一变，全部文件立刻重建，且日志能说清变化原因。
    """
    parts = [chunk_signature()]
    if embedder is not None:
        # getattr 兜底：允许传入未实现 identity 的第三方 Embedder（例如测试替身），
        # 此时退化成用类名——虽然不够精确，但至少"换类"会被识别出来。
        parts.append(str(getattr(embedder, "identity", type(embedder).__name__)))
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def combine_fingerprint(file_hash: str, signature: str) -> str:
    """把「文件内容哈希」与「索引签名」合成一个可比较的指纹串。

    格式 `内容哈希.索引签名`（点号是刻意选的：两侧都是十六进制，
    不含点号，因此可以安全地 split 回来做"变化归因"）。
    """
    return f"{file_hash}.{signature}"


def fingerprint_reason(stored: Optional[str], file_hash: str, signature: str) -> str:
    """解释"这个文件为什么需要重建"——用于日志与可视化面板的归因。

    不区分原因的话，用户只能看到 `children=N` 变多，无法判断
    到底是"我改了文档"还是"我改了参数"（两者该采取的后续动作完全不同）。
    """
    if not stored:
        return "首次索引"
    parts = stored.split(".")
    if len(parts) != 2:
        return "旧格式指纹（不含索引签名）→ 重建一次以升级"
    stored_hash, stored_sig = parts
    if stored_hash != file_hash and stored_sig != signature:
        return "文件内容与索引签名（切分参数/嵌入模型）均已变化"
    if stored_hash != file_hash:
        return "文件内容已变化"
    return "文件内容未变，但索引签名已变化（改动的是切分参数或嵌入模型）"


class RAGPipeline:
    """所有依赖从构造参数注入，因此换任意一层都不需要改本文件的代码。

    构造参数就是三个"可替换点"：
        embedder   换模型（如 HashEmbedder → OpenAIEmbedder）
        store      换向量库（如 MemoryStore → PgVectorStore）
        generator  换 LLM（如 EchoGenerator → OpenAIGenerator），也可不传（只做检索）
    加上两个开关（use_bm25 / use_mmr），恰好构成评测消融实验的四个维度。
    """

    def __init__(self, embedder: Embedder, store: VectorStore,
                 use_bm25: bool = True, use_mmr: bool = True,
                 generator: Optional[Generator] = None,
                 trace: Optional[Callable[..., None]] = None):
        self.embedder, self.store = embedder, store
        self.use_bm25, self.use_mmr = use_bm25, use_mmr
        self.generator = generator
        # BM25 索引是进程内的状态（它需要全量语料算 IDF，无法逐条增量维护）。
        # 因此它在 index() 之后重建一次，之后所有查询复用这一份。
        self._bm25: Optional[BM25] = None
        # 伪向量（HashEmbedder）的余弦分与真实模型差一个数量级，
        # 按绝对阈值做支持度门禁会把离线演示判成永久「资料不足」——
        # 因此对这类 embedding 自动降级为「不按支持度拒答」，并保留空上下文门禁。
        self.support_gate = (not getattr(embedder, "pseudo_vectors", False))
        # 观测钩子：默认空函数。可视化面板与调试工具注入实现后，
        # 每个阶段会回调一次（阶段名 / 条数 / 耗时毫秒 / 细节），
        # 用于画出"一次查询的全身剖面"。默认路径下**没有任何额外开销**。
        self._trace: Callable[..., None] = trace if trace is not None else _noop_trace
        # 索引签名在构造时算一次：把切分参数与嵌入模型身份"冻结"成指纹的一部分。
        self.index_signature = compute_index_signature(embedder)

    def _emit(self, stage: str, ms: float, count: Optional[int] = None,
              detail: Optional[Dict] = None) -> None:
        """把一次阶段的观测数据交给钩子。

        钩子内部抛错**不得影响检索本身**：可视化层的一个 bug 不应该让问答挂掉。
        因此这里吞掉异常并降级为一条 debug 日志（而不是让整个管道失败）。
        """
        try:
            self._trace(stage=stage, count=count, ms=ms, detail=detail or {})
        except Exception:                              # noqa: BLE001
            logger.debug("trace 钩子异常（已忽略，不影响检索）", exc_info=True)

    # ---------------- 离线：索引 ----------------
    def index(self, path: str, sync: bool = True) -> Dict:
        """建立或更新索引。

        指纹由 Store 持久化。真实存储模式下重启进程也能跳过未变文件；
        sync=True 时会删除该目录中已经消失的文件对应的旧索引。

        返回 parents / children / skipped / deleted / empty：
        `empty` 是「切不出任何子块」的文件数（空文件、纯符号、超短文件）。
        这类文件也会登记指纹——否则每次 index() 都要重新读取与切分它们。

        注意这里"每个文件独立处理"的设计：
        每个文件都单独完成"切分 → 嵌入 → 写入"，而不是先全部切分再统一嵌入。
        好处是单文件粒度的事务边界清晰（store.replace_document 一个文件一次），
        失败时不会出现"整个库半新半旧"。
        """
        n_parent = n_child = skipped = empty = 0
        # 归一化成绝对路径：Store 里存的 source 也是绝对路径，
        # 后面据此判断"哪些已索引文件在磁盘上消失了"。
        root = os.path.abspath(path)
        sources = set()                                   # 本轮在磁盘上实际看到的文件
        # 把签名打进索引日志：出问题时第一眼就能确认"这次索引用的是哪套参数/哪个模型"。
        logger.debug("索引签名=%s（切分参数 %s + 嵌入模型 %s）",
                     self.index_signature, chunk_signature(),
                     getattr(self.embedder, "identity", type(self.embedder).__name__))
        for doc in load_converted(path):
            src = doc.path
            sources.add(src)
            # 依赖缺失等"不可用"情况：记录原因并跳过，且**不登记指纹**
            # （下次索引还会重试，避免因为一次环境问题把文件永久标记为已处理）
            if not doc.available:
                for message in doc.warnings:
                    logger.warning("%s", message)
                continue
            for message in doc.warnings:
                logger.info("%s", message)                 # 转换为空 / 未识别标题等提示
            # ★ 增量索引的核心：指纹 = 内容哈希 + 索引签名，两者都没变才整篇跳过。
            #   这一步省掉的是**真金白银**——跳过意味着不转换、不切分、不重新 embedding。
            #   用内容哈希而不是修改时间：复制/checkout 会改时间但内容没变。
            #   掺入索引签名则解决另一半问题：内容没变、但切分参数或嵌入模型变了，
            #   此时**必须重建**，否则新参数永远不会生效（详见 compute_index_signature）。
            expected = combine_fingerprint(doc.fingerprint, self.index_signature)
            stored = self.store.get_document_fingerprint(src)
            if stored == expected:
                skipped += 1
                logger.debug("跳过未变动文件：%s", src)
                continue
            logger.debug("重建 %s：%s", os.path.basename(src),
                         fingerprint_reason(stored, doc.fingerprint, self.index_signature))
            # source_label 是面包屑兜底：txt / 去标签后的 HTML 自身没有标题，
            # 用文件名当根节点，检索结果才能溯源到文件层级（否则 heading 为空）
            idx = build_parent_child(doc.markdown, src, source_label=doc.label)
            parents, children = idx["parents"], idx["children"]
            if not children:
                # ★ 即使没有子块，也要登记指纹（空文档 → parents/children 均为空）。
                #   原实现直接 continue，导致空文件每次都被重新读取+切分，
                #   且统计上还谎报为 skipped——既浪费又掩盖真相。
                self.store.replace_document(src, expected, [], [])
                empty += 1
                logger.info("文件切不出子块（已登记指纹）：%s", src)
                continue
            # 只把子块文本送去 embedding（父块不参与检索，不需要向量）
            vecs = embed_all(self.embedder, [c["child_text"] for c in children])
            # zip 把向量绑回对应的子块。embed_all 保证顺序与输入一致，
            # 所以这里可以放心按下标配对；一旦顺序错位，向量就会挂到错误的内容上。
            for c, v in zip(children, vecs):
                c["embedding"] = v
                c["source"] = src                         # 冗余存一份，便于子块直接溯源
            self.store.replace_document(src, expected, list(parents.values()), children)
            n_parent += len(parents)
            n_child += len(children)
            logger.info("已索引 %s：父块 %d，子块 %d", os.path.basename(src),
                        len(parents), len(children))
        deleted = 0
        if sync:
            # 目录同步：把"索引里有、但磁盘上已不存在"的文件清掉。
            # 为什么需要：文件被删除或改名后，如果不清理，检索仍会返回已删除文档的内容
            # （用户会看到"资料里有这段"但去找文件却找不到）。
            for source in self.store.list_sources(root):
                if source not in sources:                 # 本轮没扫到 = 磁盘上没了
                    self.store.delete_document(source)
                    deleted += 1
                    logger.info("清理已删除文件的索引：%s", source)
        self._rebuild_bm25()
        logger.debug("索引完成：父块 %d，子块 %d，跳过 %d，删除 %d，空文件 %d",
                     n_parent, n_child, skipped, deleted, empty)
        return {"parents": n_parent, "children": n_child,
                "skipped": skipped, "deleted": deleted, "empty": empty,
                # 一并回传签名：可视化面板要显示"本次索引用的是哪套参数/模型"，
                # 并据此判断"磁盘文件数 == 库内文件数"是否成立。
                "signature": self.index_signature}

    def _rebuild_bm25(self):
        """重建关键词索引。必须在所有文件写完之后做一次——不能边索引边重建。

        原因：IDF 依赖"全库文档数"，语料还在变化时算出的 IDF 是错的
        （每新增一个文件，所有词的 IDF 都会变）。所以只能在索引全部结束后统一算。
        """
        if not self.use_bm25:
            return                                        # 消融实验里关掉 BM25 时直接跳过
        rows = self.store.list_children()
        if not rows:
            # 空语料：置空而不是留着一个基于旧语料的索引（否则会检索到已删除内容）
            self._bm25 = None
            return
        self._bm25 = BM25().fit(rows)
        logger.debug("BM25 索引重建完成：%d 个子块", len(rows))

    # ---------------- 在线：检索 ----------------
    def retrieve(self, query: str, top_k: int = C.TOP_K,
                 history_tokens: int = 0, min_sim: Optional[float] = None) -> Dict:
        """只做检索与上下文装填，不渲染 prompt、不调用 LLM。

        query() 与 answer() 都走这里，保证两条路径的检索行为**完全一致**——
        否则「评测检索用 query、线上用 answer」会得到两套不同的召回结果。
        """
        # 查询也要用**同一个 embedder** 变成向量：两边的向量空间必须一致，
        # 否则余弦相似度毫无意义（换模型时必须同时重算所有子块向量，原因就在这里）
        t0 = time.perf_counter()
        qv = embed_all(self.embedder, [query])[0]
        self._emit("embed.query", (time.perf_counter() - t0) * 1000, 1,
                   {"embedder": getattr(self.embedder, "identity", "?")})
        # "先多召回，再精排"：候选数取 top_k 的若干倍。
        # 为什么不能只召回 top_k：重排（MMR）和聚合都会淘汰一部分候选，
        # 若一开始只取 top_k，淘汰后就凑不满最终要的条数了。
        cand = top_k * C.CANDIDATE_MUL
        # ⚠ MIN_SIM 的合理取值依赖 embedding 模型：ada-002 常取 0.35，
        #   但 HashEmbedder（演示用伪向量）分数普遍偏低，需传 0 关闭过滤
        threshold = C.MIN_SIM if min_sim is None else min_sim

        # 通道 ①：向量召回（语义相近）
        t0 = time.perf_counter()
        vec_hits = self.store.search_children(qv, top_k=cand, min_sim=threshold)
        self._emit("recall.vector", (time.perf_counter() - t0) * 1000, len(vec_hits),
                   {"min_sim": threshold, "candidates": cand,
                    "top_score": vec_hits[0]["score"] if vec_hits else None})
        logger.debug("查询 %r：向量召回 %d 条（阈值 min_sim=%s，候选 %d）",
                     query, len(vec_hits), threshold, cand)

        kw_hits: list = []
        if self.use_bm25 and self._bm25:
            # 通道 ②：关键词召回（字面精确）。传入的是**原始查询字符串**，
            # 而不是向量——BM25 需要的是词，不是向量。
            t0 = time.perf_counter()
            kw_hits = self._bm25.search(query, top_k=cand)
            self._emit("recall.keyword", (time.perf_counter() - t0) * 1000, len(kw_hits),
                       {"top_score": kw_hits[0]["score"] if kw_hits else None})
            # 融合：两路各取所长。若某一路为空，rrf_fuse 仍能正常工作
            # （那一列不贡献分数，等价于退化成单通道）
            t0 = time.perf_counter()
            fused = rrf_fuse([vec_hits, kw_hits])       # 两路融合
            self._emit("fuse.rrf", (time.perf_counter() - t0) * 1000, len(fused),
                       {"channel_sizes": [len(vec_hits), len(kw_hits)]})
            logger.debug("关键词通道 %d 条，融合后 %d 条", len(kw_hits), len(fused))
        else:
            fused = vec_hits

        t0 = time.perf_counter()
        agg = aggregate_by_parent(fused)                # ★ 先聚合，后截断
        agg = agg[:top_k * C.CANDIDATE_MUL]
        # 注释再强调一次顺序：如果调换成"先截断再聚合"，top_k=3 时取到的前 3 条
        # 可能全属于同一个父块，聚合后只剩 1 条结果。
        self._emit("aggregate.parent", (time.perf_counter() - t0) * 1000, len(agg),
                   {"before": len(fused), "kept": len(agg)})

        # 向量打分统一在这一层取：MMR 需要子块向量做多样性计算，
        # 拒答门禁需要原始余弦分做支持度判断——一次取用、两处复用，避免重复查库。
        vectors = self.store.get_child_vectors([h["child_id"] for h in agg])

        # 重排前留一份快照：可视化面板要展示"MMR 把顺序改成了什么样"，
        # 只看最终顺序无法判断重排到底起了作用还是把好结果挤掉了。
        agg_pre_rerank = list(agg)

        t0 = time.perf_counter()
        if self.use_mmr:                                # 多样性重排（内部截断到 top_k）
            agg = mmr(agg, vectors, top_k=top_k)
        else:
            agg = agg[:top_k]                           # ⚠ 不开 MMR 也要截断，否则返回候选数
            # 这一行是真实踩过的坑：不开 MMR 时 agg 还停留在"候选数"（top_k×4），
            # 于是关掉重排反而返回了 4 倍的条数。截断是两种分支都必须做的收尾。
        self._emit("rerank.mmr" if self.use_mmr else "truncate",
                   (time.perf_counter() - t0) * 1000, len(agg),
                   {"enabled": self.use_mmr, "top_k": top_k})

        # 回表：命中的是子块，但喂给 LLM 的是父块全文（上下文更完整）。
        # 这是一次批量查询，避免 N+1。
        parents = self.store.get_parents([h["parent_id"] for h in agg])
        t0 = time.perf_counter()
        packed = pack_context(agg, parents, query, history_tokens)
        self._emit("context.pack", (time.perf_counter() - t0) * 1000, len(packed["blocks"]),
                   {"budget": packed["budget"], "used_tokens": packed["used_tokens"],
                    "utilization": packed["utilization"], "dropped": len(packed["dropped"])})

        # 向量支持度在**融合与 MMR 之前**用原始向量召回结果计算，理由见 answer()。
        vec_vectors = {cid: v for cid, v in
                       self.store.get_child_vectors([h["child_id"] for h in vec_hits]).items()}
        t0 = time.perf_counter()
        vec_support = support_score(vec_hits, vec_vectors, qv)
        self._emit("support.score", (time.perf_counter() - t0) * 1000, len(vec_hits),
                   {"vec_support": vec_support, "gate": self.support_gate,
                    "threshold": C.REFUSE_MIN_VEC_SCORE})

        logger.debug("上下文装填：%d 块 / %d token（利用率 %.0f%%），丢弃 %d 块，支持度 %.4f",
                     len(packed["blocks"]), packed["used_tokens"],
                     packed["utilization"] * 100, len(packed["dropped"]), vec_support)
        if packed["dropped"]:
            # dropped 是"预算不足"的信号，值得一条 INFO 级日志：
            # 若长期出现，说明该调小父块尺寸或提高预算（而不是静默丢资料）
            logger.info("有 %d 个候选块因预算不足被丢弃（top_k=%d；可考虑调小父块或提高预算）",
                        len(packed["dropped"]), top_k)

        return {
            "query": query,
            "hits": agg,                 # 最终命中的子块（已截断到 top_k）
            "agg_hits": agg_pre_rerank,  # 重排/截断**之前**的聚合结果（看重排做了什么）
            "vec_hits": vec_hits,        # 融合前的向量通道结果（原始余弦分在此）
            # 关键词通道原始结果：可视化面板要并排对比"两路各找到了什么"，
            # 只有融合后的结果看不出"是谁把这一条捞上来的"。
            "kw_hits": kw_hits,
            "vectors": vectors,          # child_id → 子块向量（供支持度/调试复用）
            "query_vec": qv,
            "vec_support": vec_support,  # 拒答门禁用的"支持度"
            "packed": packed,            # 已装填、已编号的上下文
            "parents_used": parents,     # 命中父块全文（按 parent_id 索引）
        }

    def query(self, query: str, top_k: int = C.TOP_K,
              history_tokens: int = 0, min_sim: Optional[float] = None) -> Dict:
        """检索 + 渲染 prompt。不调用 LLM，行为与加入生成层之前保持一致。

        用于两类场景：
            · 调用方想自己接管生成（用别的模型、别的框架）；
            · 评测检索指标（**不花钱、完全可复现**，因此适合当回归门禁）。
        """
        result = self.retrieve(query, top_k=top_k,
                               history_tokens=history_tokens, min_sim=min_sim)
        # strict 保持默认 False：这样 query() 的 prompt 与历史版本逐字一致，
        # 依赖它的 stress_test 等外部调用不会因为生成层的加入而行为漂移
        result["prompt"] = render_prompt(result["packed"], query)
        return result

    def answer(self, query: str, top_k: int = C.TOP_K,
               history_tokens: int = 0, min_sim: Optional[float] = None,
               strict_citations: bool = C.USE_CITATION_CONSTRAINT) -> Dict:
        """检索 + 生成 + 引用回填校验 + 拒答。

        比 query() 多做四件事：
            1. 用严格系统提示词渲染 prompt（强制 [n] 引用 + 资料不足时明说）；
            2. 生成前拒答门禁（无资料 / 向量支持度不足 → 不调用 LLM）；
            3. 生成后引用回填校验（凭空编号、无据结论可被检出）；
            4. 把引用编号解析成可溯源的出处清单。

        注意 2、3、4 都不在这个方法里实现，而是封装在 generator.answer() 里。
        这样做的好处：换 LLM 供应商（OpenAI → 别的）时，拒答策略与引用校验
        **不需要重新实现**——它们属于"编排策略"，与具体模型无关。

        ⚠ 「向量支持度」在 retrieve() 里就已经算好（vec_support），取的是
           融合与 MMR **之前**、向量召回候选的最高原始余弦相似度：
           · 不能用 RRF 融合分——量级 ≈0.016，与阈值 0.35 差两个数量级（同 MMR 量纲事故）；
           · 不能在 MMR 之后取——MMR 为多样性会选中低相关但内容重复的块，
             最高分可能已被交换掉，支持度会被系统性低估；
           · 不能取 top-1 命中的分——top-1 未必是向量分最高的那个候选。
           一句话：**支持度是"这轮召回到的候选里最好有多好"，而不是"最终排序第一有多好"**。
        """
        if self.generator is None:
            # 明确报错而不是返回一个空答案：调用方写错了（忘了注入 generator），
            # 静默失败会让人以为是"检索没结果"，从而排查错方向
            raise ValueError("未注入 generator：构造 RAGPipeline(generator=...) "
                             "或改用 query() 只取 prompt")

        result = self.retrieve(query, top_k=top_k,
                               history_tokens=history_tokens, min_sim=min_sim)

        # strict=True：注入 config.ANSWER_SYSTEM_PROMPT，里面明确要求
        # "只依据参考资料作答"与"每个事实性陈述标注 [n]"。
        # 这是三层防幻觉防线里的第二层（生成中约束）。
        prompt = render_prompt(result["packed"], query, strict=strict_citations)
        # support_gate=False 时传 None：Generator 会跳过 weak_support 判定，
        # 但「无资料」门禁与生成后的引用校验照常生效。
        answer = self.generator.answer(
            result["packed"], prompt, question=query,
            vec_score=result["vec_support"] if self.support_gate else None,
            strict_citations=strict_citations)
        result["prompt"] = prompt
        result["answer"] = answer
        return result
