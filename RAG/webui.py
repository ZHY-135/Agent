# -*- coding: utf-8 -*-
"""⑩ 可视化观测台：把"一次提问"在整条流水线里的中间结果画出来。

--------------------------------------------------------------------------------
为什么需要它
--------------------------------------------------------------------------------
命令行只能看到**最终结果**（prompt / 答案）。当结果不对时，最常问的三个问题是：

    · 是向量没召回，还是关键词没召回？（两路通道各自找到了什么）
    · 是先聚合后截断的顺序被改坏了吗？（聚合前后条数）
    · 是上下文预算把正确答案挤掉了吗？（每块占了多少 token、谁被 dropped）

`debug_trace.py` 用文本回答了这些问题（逐阶段打印），本模块把**同一份数据**
画成界面：瀑布图 + 三路榜单 + 预算条 + 拒答原因 + 评测指标。

--------------------------------------------------------------------------------
设计要点（读懂这个文件的关键）
--------------------------------------------------------------------------------
1. **数据不重算**：所有内容都来自 `pipeline.retrieve()` / `answer()` 的返回值，
   以及注入的 `TraceRecorder` 采集到的阶段耗时。观测台不复制任何检索逻辑，
   因此它显示的**必然**是真实链路的行为（不存在"面板和线上两套真相"）。
2. **观测钩子默认关闭**：`RAGPipeline(trace=None)` 时零开销；只有本模块注入。
3. **开关即时生效**：`use_bm25` / `use_mmr` 是构造参数，改开关会重建 pipeline
   与索引（这是刻意的：BM25 的 IDF 依赖全量语料，复用会让实验互相污染）。

--------------------------------------------------------------------------------
运行方式
--------------------------------------------------------------------------------
    python -m RAG.main serve                    # 推荐：自动装配参数并打开浏览器
    streamlit run RAG/webui.py                  # 等价写法
    python -m RAG.main serve --docs docs --port 8765 --no-browser
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---- 引导：streamlit 是以"脚本"方式执行本文件的（__main__），没有包上下文，
#      相对导入会失败。因此把**项目根**放进 sys.path，改用绝对导入 RAG.xxx。
#      这也是 streamlit 官方推荐的"包内应用"写法。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import pandas as pd
    import streamlit as st
except ImportError as error:                            # pragma: no cover - 依赖缺失路径
    raise SystemExit(
        "观测台需要 streamlit（含 pandas）：\n"
        "    uv pip install streamlit\n"
        "或（在实机上，把依赖写进 pyproject 的 webui 分组）：\n"
        "    uv sync --extra webui\n"
        f"原始错误：{error}") from error

from RAG import config as C
from RAG.chunkers import chunk_signature
from RAG.converters import REASON_DEGRADED, REASON_HINT, index_capability
from RAG.corpus import (
    DEFAULT_STATE,
    CorpusStatus,
    add_favorite,
    check_corpus_dir,
    favorite_rows,
    load_ui_state,
    native_pick_directory,
    normalize_dir,
    remove_favorite,
    save_ui_state,
    state_to_json,
    supported_suffixes_text,
)
from RAG.embedders import CachingEmbedder, HashEmbedder
from RAG.evaluation import (
    ABLATIONS,
    check_anchors,
    load_golden_set,
    render_check_report,
    render_report,
    run_experiment,
)
from RAG.generators import EchoGenerator
from RAG.loaders import file_fingerprint, iter_documents
from RAG.pipeline import (
    RAGPipeline,
    combine_fingerprint,
    compute_index_signature,
    fingerprint_reason,
)
from RAG.stores import MemoryStore

DEFAULT_DOCS = "docs"
DEFAULT_QUESTION = "连接池最大连接数是多少"


def _cli_docs_given() -> Optional[str]:
    """读取 `python -m RAG.main serve --docs X` 传进来的语料目录（没传则 None）。

    streamlit 会把 `--` 之后的参数原样交给应用脚本的 sys.argv，
    因此这里做一次极简解析（只认 --docs）——不引入 argparse 是为了避免
    streamlit 自己的参数与我们的参数在同一份 argv 里互相报错。

    为什么区分"传了"与"没传"：命令行是**这一次的显式意图**，应当压过记忆里的设置；
    没传时才回退到上次保存的选择。否则 `serve --docs A` 会被旧状态静默覆盖。
    """
    argv = sys.argv
    if "--docs" in argv:
        index = argv.index("--docs")
        if index + 1 < len(argv):
            return argv[index + 1]
    return None


# ============================== 观测记录器 ==============================

class TraceRecorder:
    """收集一次查询内各阶段的观测数据（作为 pipeline 的 trace 钩子注入）。

    为什么做成"有状态的可调用对象"而不是一个回调函数：
      pipeline 在构造时就固定了钩子，而每次查询要重新收集。
      把"当前这一批记录"放在对象内部、查询前 reset 一次，
      就不需要为了换钩子而重建 pipeline（也就不会连带重建索引）。
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def __call__(self, *, stage: str, count: Optional[int], ms: float,
                 detail: Optional[Dict[str, Any]] = None) -> None:
        self.records.append({"阶段": stage, "条数": count,
                             "耗时(ms)": round(ms, 2), **(detail or {})})

    def reset(self) -> None:
        self.records = []


class Session:
    """把 pipeline 与它的观测记录器绑在一起，供 `st.cache_resource` 缓存。

    为什么不直接把 recorder 挂在 pipeline 上：那需要写 pipeline 的私有属性。
    缓存一个"配对对象"更干净，也避免了两者生命周期不一致（面板重启但 pipeline 还在）。

    ⚠ recorder 必须**由外部注入**（就是当初传给 RAGPipeline(trace=...) 的那一个）：
      本类如果自己 new 一个，就会变成"面板清空的是 A、pipeline 往 B 里写"，
      表现为瀑布表永远为空——而且**不会报任何错**。
    """

    def __init__(self, pipeline: RAGPipeline, docs_dir: str, recorder: TraceRecorder) -> None:
        self.pipeline = pipeline
        self.recorder = recorder
        self.docs_dir = docs_dir
        self.last_index: Dict[str, Any] = {}
        # 语料目录的校验结论（由 get_session 填入，界面直接渲染）
        self.corpus_status: CorpusStatus = CorpusStatus()
        # ---- 多轮追问（4A）----
        # 历史挂在 Session 上而不是 st.session_state：Session 由 cache_resource 缓存，
        # 与 pipeline 同生命周期，不会出现"pipeline 换了但历史还在"的错配。
        self.turns: List[Dict[str, str]] = []
        self.multi_turn: bool = False
        # 去重用：面板每次交互都会整页重跑，没有它同一轮会被反复记录（见 record_turn）
        self.recorded_question: str = ""


# ============================== 装配（缓存） ==============================

@st.cache_resource(show_spinner=False)
def get_session(docs_dir: str, use_bm25: bool, use_mmr: bool,
                with_generator: bool) -> Session:
    """构建并索引一个 pipeline；参数变化时自动重建（Streamlit 缓存键即参数）。

    ⚠ 这里**刻意不复用 store**：`use_bm25` / `use_mmr` 决定索引构建方式，
    复用会让上一组实验的状态漏进下一组（真实踩过的坑：关掉 BM25 的那组
    仍命中大量关键词结果，因为 BM25 索引还在）。

    ⚠ 目录无效时**不抛异常**：语料目录是用户随手输入的，打错一个字很常见。
    校验交给 `corpus.check_corpus_dir()`，这里只负责"能用就建索引，不能用就空着"，
    由界面把原因说清楚——否则整页会变成红色异常栈，看起来像程序坏了。
    """
    recorder = TraceRecorder()
    pipeline = RAGPipeline(
        # 缓存装饰器只做"内容→向量"的映射，复用安全且有收益
        CachingEmbedder(HashEmbedder()),
        MemoryStore(),
        use_bm25=use_bm25,
        use_mmr=use_mmr,
        generator=EchoGenerator() if with_generator else None,
        trace=recorder,
    )
    session = Session(pipeline, docs_dir, recorder)
    status = check_corpus_dir(docs_dir)
    if status.usable:
        session.last_index = pipeline.index(docs_dir)
    session.corpus_status = status
    return session


# ============================== 展示辅助 ==============================

def _fmt_score(score: Any) -> str:
    return f"{score:.4f}" if isinstance(score, (int, float)) else "-"


def hits_rows(hits: Sequence[Dict], limit: int = 10) -> List[Dict[str, Any]]:
    """把命中列表转成表格行。**量纲一并显示**——这是本项目的核心易错点：
    余弦、BM25、RRF 三种分数量级完全不同，混在一起人眼会误判。"""
    rows = []
    for rank, hit in enumerate(list(hits)[:limit], start=1):
        rows.append({
            "排名": rank,
            "分数": _fmt_score(hit.get("score")),
            "量纲": hit.get(C.SCORE_KIND_KEY, "-"),
            "来源": Path(str(hit.get("source") or "")).name,
            "面包屑": hit.get("heading") or "（无）",
            "片段": (hit.get("child_text") or hit.get("text") or "")[:80].replace("\n", " "),
        })
    return rows


def corpus_rows(session: Session) -> List[Dict[str, Any]]:
    """语料台账：磁盘上有什么、库里有没有、是否需要重建（以及为什么）。

    ★ 这张表是"静默错误"的照妖镜：把索引签名与文件指纹并排显示后，
      "改了切分参数却没重建"这类问题会直接暴露成一行「需重建：索引签名已变化」。
    """
    pipeline, store = session.pipeline, session.pipeline.store
    signature = pipeline.index_signature
    rows = []
    try:
        files = list(iter_documents(session.docs_dir))
    except (FileNotFoundError, ValueError) as error:
        return [{"文件": f"⚠ 语料目录不可用：{error}"}]
    # 子块只取一次并按文件名归组：面板每次交互都会重跑本函数，
    # 逐文件调用 store.list_children() 会退化成 O(文件数 × 块数)。
    children_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for child in store.list_children():
        name = Path(str(child.get("source") or "")).name
        children_by_name.setdefault(name, []).append(child)
    for path in files:
        name = Path(path).name
        ok, reason = index_capability(path)
        if not ok:
            # ⛔ 这类文件**永远不会**被索引，所以绝不能显示成"需重建"——
            #    那会让用户以为"等一等或点一下就 好了"，于是一直等下去。
            #    台账的职责正是把这种"永远等不到结果"的状态说穿。
            rows.append({
                "文件": name,
                "大小(KB)": round(Path(path).stat().st_size / 1024, 1),
                "子块": 0,
                "父块": 0,
                "内容指纹": "-",
                "库内状态": f"⛔ 不参与索引：{REASON_HINT.get(reason, reason)}",
            })
            continue
        fingerprint = file_fingerprint(path)
        stored = store.get_document_fingerprint(path)
        if stored == combine_fingerprint(fingerprint, signature):
            status = "✅ 已索引（内容与参数均未变）"
        else:
            status = "🔁 需重建：" + fingerprint_reason(stored, fingerprint, signature)
        if reason == REASON_DEGRADED:
            # 降级索引也要说清：内容搜得到，但结构（标题层级）有损，
            # 否则用户会觉得"明明索引了，为什么溯源只能到文件名"
            status += "（⚠ 降级：结构有损）"
        children = children_by_name.get(name, [])
        rows.append({
            "文件": name,
            "大小(KB)": round(Path(path).stat().st_size / 1024, 1),
            "子块": len(children),
            "父块": len({c.get("parent_id") for c in children}),
            "内容指纹": fingerprint[:8],
            "库内状态": status,
        })
    return rows


def component_rows(session: Session) -> List[Dict[str, Any]]:
    pipeline = session.pipeline
    embedder = pipeline.embedder
    # ⚠ "维度" 列刻意统一成字符串：混入 int 会让 pyarrow 推断列类型失败
    #   （实测报 "Could not convert '-' with type str: tried to convert to int64"），
    #   表格会退化成"自动修复类型"的告警路径。
    return [
        {"组件": "嵌入器", "实现": getattr(embedder, "identity", type(embedder).__name__),
         "维度": str(getattr(embedder, "dim", "-")),
         "说明": "伪向量（仅回归对比）" if getattr(embedder, "pseudo_vectors", False)
                 else "真实语义向量"},
        {"组件": "向量库", "实现": type(pipeline.store).__name__, "维度": "-",
         "说明": "内存实现（演示）" if isinstance(pipeline.store, MemoryStore)
                 else "pgvector（生产）"},
        {"组件": "关键词通道", "实现": "BM25" if pipeline.use_bm25 else "已关闭",
         "维度": f"{len(pipeline.store.list_children())} 块",
         "说明": "字面精确匹配，专治错误码/API 名" if pipeline.use_bm25 else "消融实验：纯向量"},
        {"组件": "重排", "实现": "MMR" if pipeline.use_mmr else "已关闭",
         "维度": f"λ={C.MMR_LAMBDA}", "说明": "多样性去冗（不是相关性精排）"},
        {"组件": "生成器", "实现": type(pipeline.generator).__name__ if pipeline.generator
                                     else "未注入",
         "维度": "-",
         "说明": "离线复述（验证引用链路）" if isinstance(pipeline.generator, EchoGenerator)
                 else "未接生成层：只能看检索"},
        {"组件": "索引签名", "实现": pipeline.index_signature, "维度": chunk_signature(),
         "说明": "切分参数 + 嵌入模型的组合标识；变了就必须重建索引"},
    ]


def chunk_param_rows() -> List[Dict[str, Any]]:
    """当前生效的切分参数——面板里直接看到"这次索引用的到底是哪套参数"。"""
    return [
        {"参数": "MAX_PARENT_CHARS", "值": C.MAX_PARENT_CHARS, "作用": "父块上限（喂给 LLM 的粒度）"},
        {"参数": "MAX_CHILD_CHARS", "值": C.MAX_CHILD_CHARS, "作用": "子块上限（参与 embedding 的粒度）"},
        {"参数": "CHILD_OVERLAP", "值": C.CHILD_OVERLAP, "作用": "滑窗重叠（防止答案被切断）"},
        {"参数": "MIN_CHILD_CHARS", "值": C.MIN_CHILD_CHARS, "作用": "过短尾块的合并阈值"},
        {"参数": "MERGE_RATIO", "值": C.MERGE_RATIO, "作用": "合并后允许超出上限的倍数"},
        {"参数": "TOP_K", "值": C.TOP_K, "作用": "最终返回条数"},
        {"参数": "CANDIDATE_MUL", "值": C.CANDIDATE_MUL, "作用": "候选倍数（先多召回再精排）"},
        {"参数": "MIN_SIM", "值": C.MIN_SIM, "作用": "余弦相似度下限（伪向量需传 0）"},
        {"参数": "RRF_K", "值": C.RRF_K, "作用": "RRF 融合常数"},
        {"参数": "MMR_LAMBDA", "值": C.MMR_LAMBDA, "作用": "相关性 vs 多样性"},
        {"参数": "CONTEXT_BUDGET", "值": C.CONTEXT_BUDGET, "作用": "prompt token 预算"},
        {"参数": "REFUSE_MIN_VEC_SCORE", "值": C.REFUSE_MIN_VEC_SCORE,
         "作用": "拒答阈值（只与原始余弦比较！）"},
    ]


# ============================== 各标签页 ==============================

def tab_overview(session: Session, docs_dir: str) -> None:
    pipeline = session.pipeline
    stat = session.last_index
    status = session.corpus_status
    st.markdown(f"**当前语料**：`{status.path or docs_dir}` — {status.icon} {status.message}")

    if not status.usable:
        st.warning("语料目录不可用，索引为空。请在左侧「📁 语料目录」里修正路径——"
                   "相对路径按项目根解析，也可以直接填绝对路径。")
        return
    left, mid, right = st.columns(3)
    # 文件数直接用校验阶段已经算好的结果，避免在这里再扫一遍目录
    left.metric("语料文件", status.n_files)
    mid.metric("父块 / 子块", f"{len({c['parent_id'] for c in pipeline.store.list_children()})}"
                             f" / {len(pipeline.store.list_children())}")
    right.metric("上次索引", f"新建 {stat.get('parents', 0)} 父块 / 跳过 {stat.get('skipped', 0)}")

    # ★ "文件数 ≠ 会被索引的文件数"。这一段存在的唯一理由：
    #   没有它时，用户把 PDF 拖进语料目录后看到的是"语料文件 3 / 父块 0"，
    #   要在"路径错了 / 程序坏了 / 格式不支持"之间猜。现在直接点名原因。
    if status.unsupported:
        st.warning(f"有 {status.n_files - status.indexable} 个文件**不会进入索引**："
                   f"{status.unsupported_text}。这些文件不计入下面的父块/子块统计。")
    if status.degraded:
        st.info(f"有 {status.degraded} 个文件为**降级索引**：内容搜得到，但标题层级丢失、"
                f"面包屑会退化为文件名。")
    if stat.get("unsupported"):
        st.caption(f"上次索引实际跳过了 {stat['unsupported']} 个文件："
                   f"{'、'.join((stat.get('unsupported_files') or [])[:5])}")

    st.caption("组件装配（换任意一层都只改 main.py 的装配函数，编排层零改动）")
    st.dataframe(pd.DataFrame(component_rows(session)), hide_index=True, width="stretch")

    st.subheader("语料台账")
    st.caption("库内状态由「内容哈希 + 索引签名」共同判定——参数或模型变了会显示为「需重建」，"
               "这是本项目修掉的一个静默错误：改切分参数后旧实现会谎报「已跳过」。")
    st.dataframe(pd.DataFrame(corpus_rows(session)), hide_index=True, width="stretch")

    col_a, col_b = st.columns([1, 3])
    if col_a.button("🔄 重建索引", width="stretch"):
        with st.spinner("索引中…"):
            session.last_index = pipeline.index(docs_dir)
        st.success(f"索引完成：{session.last_index}")
    col_b.caption("未改动的文件会被跳过（省 embedding 费用）；改了参数或模型则全部重建。")

    with st.expander("当前生效的切分与检索参数（config.py）"):
        st.dataframe(pd.DataFrame(chunk_param_rows()), hide_index=True, width="stretch")


def tab_retrieval(session: Session, question: str, top_k: int, min_sim: float) -> Dict:
    pipeline, recorder = session.pipeline, session.recorder
    # 多轮模式：把历史交给 pipeline，由它统一负责"渲染历史 + 扣预算"。
    # 关掉时传 None —— 这样 prompt 与单轮模式逐字一致，方便对照着看差异。
    history = session.turns if session.multi_turn else None
    recorder.reset()
    result = pipeline.query(question, top_k=top_k, min_sim=min_sim, history=history)

    st.subheader("① 流水线瀑布（每阶段的条数与耗时）")
    st.caption("条数下降的位置就是「淘汰」发生的地方：召回 → 融合 → 聚合 → 重排 → 装填。")
    records = recorder.records
    if records:
        frame = pd.DataFrame(records)
        st.dataframe(frame[["阶段", "条数", "耗时(ms)"]], hide_index=True, width="stretch")
        st.bar_chart(frame.set_index("阶段")["耗时(ms)"], height=180)

    st.subheader("② 两路通道 vs 融合后（看清「是谁捞上来的」）")
    col_vec, col_kw, col_fused = st.columns(3)
    with col_vec:
        st.markdown(f"**向量通道**（原始余弦，共 {len(result['vec_hits'])} 条）")
        st.dataframe(pd.DataFrame(hits_rows(result["vec_hits"])), hide_index=True,
                     width="stretch")
    with col_kw:
        st.markdown(f"**BM25 关键词通道**（共 {len(result['kw_hits'])} 条）")
        st.dataframe(pd.DataFrame(hits_rows(result["kw_hits"])), hide_index=True,
                     width="stretch")
    with col_fused:
        st.markdown(f"**融合后最终命中**（{len(result['hits'])} 条，量纲=rrf）")
        st.dataframe(pd.DataFrame(hits_rows(result["hits"])), hide_index=True,
                     width="stretch")

    vec_ids = {h["child_id"] for h in result["vec_hits"]}
    kw_ids = {h["child_id"] for h in result["kw_hits"]}
    both = vec_ids & kw_ids
    st.caption(f"两路同时命中的块：{len(both)} 个（RRF 会奖励这种「双路认可」的结果）；"
               f"仅向量命中 {len(vec_ids - kw_ids)} 个；仅关键词命中 {len(kw_ids - vec_ids)} 个。")

    st.subheader("③ 聚合与重排的先后（本项目踩过的坑）")
    before, after = len(result["agg_hits"]), len(result["hits"])
    st.markdown(f"聚合后候选 **{before}** 条 → 重排/截断后 **{after}** 条"
                f"（top_k={top_k}）。聚合**必须**发生在截断之前，否则 top_k 个名额"
                f"可能全落在同一个父块上，去重后只剩 1 条。")
    return result


def tab_context(result: Dict) -> None:
    packed = result["packed"]
    used, budget = packed["used_tokens"], packed["budget"]
    st.subheader("① 预算占用")
    left, mid, right = st.columns(3)
    left.metric("已用 / 预算", f"{used} / {budget}")
    mid.metric("利用率", f"{packed['utilization'] * 100:.0f}%")
    right.metric("被挤掉的块", len(packed["dropped"]))
    st.progress(min(1.0, packed["utilization"]))

    st.subheader("② 装进 prompt 的块（按 token 排序即实际顺序）")
    blocks = packed["blocks"]
    if blocks:
        frame = pd.DataFrame([{
            "编号": f"[{b.get('ref', i)}]",
            "类型": b.get("kind"),
            "tokens": b.get("tokens"),
            "分数": _fmt_score(b.get("score")),
            "量纲": b.get(C.SCORE_KIND_KEY) or "-",
            "来源": Path(str(b.get("source") or "")).name,
            "面包屑": b.get("heading") or "（无）",
        } for i, b in enumerate(blocks, start=1)])
        st.dataframe(frame, hide_index=True, width="stretch")
        st.bar_chart(frame.set_index("编号")["tokens"], height=180)
        with st.expander("逐块看正文（点开检查是否真的含答案）"):
            for i, block in enumerate(blocks, start=1):
                st.markdown(f"**[{block.get('ref', i)}]** "
                            f"`{Path(str(block.get('source') or '')).name}` "
                            f"› {block.get('heading') or '（无面包屑）'}")
                st.code(str(block.get("text") or "")[:1500], language="markdown")
    if packed["dropped"]:
        st.warning(f"有 {len(packed['dropped'])} 个候选块因预算不足被丢弃——"
                   f"若长期出现，应调小父块尺寸或提高 CONTEXT_BUDGET，而不是静默丢资料。")
        st.dataframe(pd.DataFrame(hits_rows(packed["dropped"])), hide_index=True,
                     width="stretch")

    st.subheader("③ 最终 prompt")
    st.code(result.get("prompt") or "（无）", language="markdown")


def record_turn(session: Session, question: str, answer_text: str) -> None:
    """把这一轮问答记进多轮历史（仅在多轮模式开启时）。

    ⚠ **必须按问题去重**：Streamlit 每次交互都会整页重跑，`tab_generation` 会被
      再次调用；不去重的话同一轮会被记录几十次，历史预算瞬间被同一句话吃光——
      表现为"明明只问了两句，为什么资料全被挤掉了"，而且完全看不出原因。
      代价是"连着问两遍同一个问题"只记一次；对追问场景这个取舍可以接受。

    只存**问题与答案正文**：prompt 和引用清单都很大，存进去会迅速吃光预算。
    """
    if not getattr(session, "multi_turn", False) or not question:
        return
    if question == session.recorded_question:
        return
    session.turns.append({"question": question, "answer": answer_text})
    session.recorded_question = question


def tab_generation(session: Session, result: Dict) -> None:
    pipeline = session.pipeline
    if pipeline.generator is None:
        st.info("当前 pipeline 未注入生成器，只能看检索。用 `--with-generator` 或 "
                "在面板顶部开启「生成层」即可看到引用校验与拒答。")
        return
    answer = pipeline.generator.answer(
        result["packed"], result.get("prompt") or "",
        question=result["query"],
        vec_score=result["vec_support"] if pipeline.support_gate else None)
    # 记录这一轮（多轮模式开关由 record_turn 内部判断，并负责按问题去重）
    record_turn(session, str(result.get("query") or ""), str(answer.get("text") or ""))

    st.subheader("① 答案")
    if answer.get("refused"):
        st.error(f"已拒答（原因码：`{answer.get('refusal_reason')}`）\n\n{answer.get('text')}")
        reason = answer.get("refusal_reason")
        st.caption({
            "no_context": "检索为空 → 生成前门禁直接拒答，**不调用** LLM（省钱也省掉一次硬编机会）。",
            "weak_support": "向量支持度低于阈值 → 说明资料与问题相关度过低。",
            "invalid_citation": "模型给出了不存在的引用编号 → 最强幻觉信号，直接拒答。",
            "not_supported": "答案没有任何合法引用 → 无据结论。",
        }.get(str(reason), "见 config.REFUSAL_MESSAGES。"))
    else:
        st.success(answer.get("text") or "")
        st.caption(f"答案 token 估算：{answer.get('answer_tokens', '-')}")

    st.subheader("② 引用回填校验（防幻觉的第三层）")
    validation = answer.get("validation") or {}
    cols = st.columns(5)
    cols[0].metric("引用数", len(validation.get("refs") or []))
    cols[1].metric("合法", len(validation.get("valid_refs") or []))
    cols[2].metric("凭空编号", len(validation.get("invalid_refs") or []))
    cols[3].metric("未使用资料", len(validation.get("unused_refs") or []))
    cols[4].metric("覆盖率", f"{(validation.get('coverage') or 0) * 100:.0f}%")
    if validation.get("invalid_refs"):
        st.error(f"⚠ 检出凭空编号：{validation['invalid_refs']} —— 这是最强的幻觉信号。")

    details = answer.get("citations") or []
    if details:
        st.markdown("**逐条溯源**（答案里的 `[n]` 对应哪一段资料）")
        st.dataframe(pd.DataFrame([{
            "引用": f"[{d.get('ref')}]",
            "来源": Path(str(d.get("source") or "")).name,
            "面包屑": d.get("heading") or "（无）",
            "块类型": "整父块" if d.get("kind") == "parent" else "子块片段",
        } for d in details]), hide_index=True, width="stretch")

    st.subheader("③ 支持度 vs 拒答阈值")
    support = float(result["vec_support"])
    threshold = C.REFUSE_MIN_VEC_SCORE
    left, right = st.columns([1, 3])
    left.metric("支持度（原始余弦）", f"{support:.4f}")
    if pipeline.support_gate:
        right.progress(min(1.0, support / max(threshold, 1e-6)))
        right.caption(f"阈值 {threshold}；低于它就会以 `weak_support` 拒答。"
                      f"⚠ 支持度必须取**原始余弦**，不能取 RRF 融合分（≈0.016 量级）。")
    else:
        right.info("当前嵌入器是伪向量（HashEmbedder），支持度门禁已自动关闭——"
                   "它的分数与真实模型差一个数量级，按 0.35 判定会让演示全部被误拒。")


def tab_eval(session: Session, docs_dir: str) -> None:
    st.caption("检索指标完全离线可复现（不调用 LLM），因此可以作为回归门禁。"
               "四组配置**各自独立建索引**，避免 BM25 状态互相污染。")
    golden = st.text_input("评测集", value="eval/smoke_golden_set.jsonl",
                           help="JSONL，# 开头为注释；reviewed:false 的题不计入指标分母")
    names = st.multiselect("消融配置", list(ABLATIONS.keys()), default=list(ABLATIONS.keys()))
    col_check, col_run = st.columns(2)

    if col_check.button("🔍 锚点自检（不跑指标）", width="stretch"):
        try:
            items = load_golden_set(golden)
        except Exception as error:                       # noqa: BLE001 - 面板需要显示原因
            st.error(f"载入评测集失败：{error}")
            return
        children = session.pipeline.store.list_children()
        parents = session.pipeline.store.get_parents([c["parent_id"] for c in children])
        for child in children:
            parent = parents.get(child["parent_id"]) or {}
            child["parent_text"] = parent.get("parent_text")
            child["heading"] = parent.get("heading")
            child["source"] = parent.get("source") or child.get("source")
        unmatched = check_anchors(items, children)
        st.markdown(render_check_report(items, unmatched, {"summary": docs_dir}))
        if unmatched:
            st.info("冒烟集里**故意**埋了一个坏锚点（q012），用来证明自检真的能发现问题。")

    if col_run.button("▶ 跑四组消融", width="stretch", type="primary"):
        try:
            items = load_golden_set(golden)
        except Exception as error:                       # noqa: BLE001
            st.error(f"载入评测集失败：{error}")
            return
        embedder = session.pipeline.embedder
        selected = [ABLATIONS[name] for name in names] or list(ABLATIONS.values())

        def factory(use_bm25: bool, use_mmr: bool) -> RAGPipeline:
            return RAGPipeline(embedder, MemoryStore(), use_bm25=use_bm25, use_mmr=use_mmr)

        with st.spinner("评测中：每组配置独立建索引…"):
            reports = run_experiment(items, docs_dir, factory, runs=selected,
                                     top_k=C.TOP_K)
        st.markdown(render_report(reports))
        rows = []
        for report in reports:
            row = {"配置": report.run.name, "计入分母": report.retrieval.n_expected}
            for k in sorted(report.retrieval.recall):
                row[f"recall@{k}"] = round(report.retrieval.recall[k], 3)
            for k in sorted(report.retrieval.mrr):
                row[f"MRR@{k}"] = round(report.retrieval.mrr[k], 3)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if len(rows) > 1:
            st.bar_chart(pd.DataFrame(rows).set_index("配置").drop(columns=["计入分母"]),
                         height=240)


# ============================== 语料管理（侧边栏） ==============================

STATE_FILE_KEY = "_rag_ui_state"


def _persist(state: Dict[str, Any]) -> None:
    """保存界面状态；失败只在侧边栏提示一次，不影响使用。"""
    st.session_state[STATE_FILE_KEY] = state
    ok = save_ui_state(state)
    st.session_state["_rag_ui_save_failed"] = not ok


def _state() -> Dict[str, Any]:
    return st.session_state.get(STATE_FILE_KEY) or dict(DEFAULT_STATE)


def _init_state() -> None:
    """初始化会话状态：命令行参数 > 上次保存的选择 > 内置默认值。

    只在第一次运行时执行（Streamlit 每次交互都会重跑整个脚本）。
    顺序刻意如此：`serve --docs A` 是"这一次的显式意图"，必须压过记忆；
    没传时才回落到上次用过的目录——否则命令行参数会被旧状态静默覆盖。
    """
    if STATE_FILE_KEY in st.session_state:
        return
    saved = load_ui_state()
    cli_docs = _cli_docs_given()
    state = dict(saved)
    state["docs_dir"] = normalize_dir(cli_docs) if cli_docs else normalize_dir(
        saved.get("docs_dir") or DEFAULT_DOCS)
    if cli_docs and not any(normalize_dir(f) == state["docs_dir"]
                            for f in saved.get("favorites") or []):
        # 命令行给的目录顺手收藏：用户显然打算用它，下次就能直接点
        state = add_favorite(state, state["docs_dir"])
    st.session_state[STATE_FILE_KEY] = state
    # 立刻落盘：否则"命令行传了 A"这一事实只活在内存里，
    # 下次启动会退回文件里的旧值（实测过：文件里留着上次输错的路径，
    # 面板却显示 A —— 两边不一致，排查时会非常费解）。
    save_ui_state(state)
    # 输入框的初值（widget 的 key 必须在这里就位，之后再改会抛异常）
    st.session_state["docs_dir_input"] = state["docs_dir"]


def _apply_favorite(path: str) -> None:
    """点「使用」时的回调：切换语料目录。

    必须用 on_click 回调（而不是在按钮返回 True 后赋值）：
    Streamlit 的回调在**下一次重跑的最开始**执行，那时输入框还没被创建，
    改 `st.session_state["docs_dir_input"]` 是合法的；
    若在按钮自己那次重跑里改，会撞上"widget 已实例化"的异常。
    """
    st.session_state["docs_dir_input"] = path
    _persist(dict(_state(), docs_dir=path))


def _add_current_favorite() -> None:
    current = normalize_dir(st.session_state.get("docs_dir_input") or "")
    if current:
        _persist(add_favorite(_state(), current))


def _reset_state() -> None:
    """恢复默认设置（并清空收藏）。"""
    fresh = dict(DEFAULT_STATE)
    st.session_state[STATE_FILE_KEY] = fresh
    st.session_state["docs_dir_input"] = DEFAULT_DOCS
    st.session_state["_rag_ui_reset_notice"] = True
    save_ui_state(fresh)


def sidebar_corpus_manager() -> CorpusStatus:
    """语料目录管理区：系统对话框 → 输入/校验 → 收藏/切换 → 返回校验结论。"""
    state = _state()
    st.header("📁 语料目录")

    # ① 系统原生「选择文件夹」对话框（像安装程序选安装路径那样，一次看到整棵树）
    if st.button("🖥 用系统对话框选择目录…", width="stretch",
                 help="弹出操作系统的文件夹选择窗口（可直接看到所有盘符与目录树）。"
                      "对话框出现在运行本面板的那台机器桌面上。"):
        with st.spinner("请在弹出的对话框里选择目录…"):
            picked = native_pick_directory(st.session_state.get("docs_dir_input") or "")
        if picked.ok:
            # 不能在这里直接改输入框的值（widget 已实例化），
            # 先寄存到 session_state，下一轮开头（widget 创建之前）再写入。
            st.session_state["_pending_docs_dir"] = picked.path
            st.session_state["_pick_notice"] = picked.message
            st.rerun()
        else:
            st.session_state["_pick_notice"] = picked.message

    notice = st.session_state.pop("_pick_notice", None)
    if notice:
        if st.session_state.get("docs_dir_input") and "已选择" in notice:
            st.success(notice)
        else:
            st.info(notice)
    st.caption("提示：对话框只会在**运行 serve 的那台机器**上弹出；"
               "若你是远程访问面板，请改用下面的浏览面板或直接粘贴路径。")

    typed = st.text_input("目录路径", key="docs_dir_input",
                          help="相对路径按项目根解析，也可填绝对路径（如 D:\\我的文档\\手册）。"
                               "目录会被递归扫描。")
    status = check_corpus_dir(typed)
    st.markdown(f"{status.icon} {status.message}")
    if status.files:
        st.caption("含：" + "、".join(status.files) + ("…" if status.n_files > len(status.files) else ""))
    if status.level == "warn":
        st.caption(f"支持的格式：{supported_suffixes_text()}")

    # 输入框变化就记住（不需要用户额外点"保存"）
    if normalize_dir(typed) != normalize_dir(str(state.get("docs_dir") or "")):
        _persist(dict(state, docs_dir=normalize_dir(typed)))

    col_add, col_reset = st.columns(2)
    already = any(normalize_dir(f) == normalize_dir(typed)
                  for f in state.get("favorites") or [])
    col_add.button("⭐ 收藏当前目录" if not already else "⭐ 已收藏",
                   on_click=_add_current_favorite, disabled=already or not typed,
                   width="stretch")
    col_reset.button("↺ 恢复默认", on_click=_reset_state, width="stretch",
                     help="清空收藏并回到内置默认值（docs）")

    favorites = [f for f in state.get("favorites") or [] if isinstance(f, str)]
    if favorites:
        st.markdown(f"**常用目录**（{len(favorites)}）")
        for row in favorite_rows(favorites):
            text_col, use_col, del_col = st.columns([5, 2, 2])
            text_col.markdown(f"{row['状态']}\n\n`{row['目录']}`")
            use_col.button("使用", key=f"use::{row['目录']}",
                           on_click=_apply_favorite, args=(row["目录"],),
                           width="stretch")
            del_col.button("移除", key=f"del::{row['目录']}",
                           on_click=lambda p=row["目录"]: _persist(remove_favorite(_state(), p)),
                           width="stretch")
    else:
        st.caption("还没有收藏。填好目录后点「⭐ 收藏当前目录」，以后一键切换。")

    if st.session_state.get("_rag_ui_save_failed"):
        st.caption("⚠️ 设置没能写入 `rag.local.json`（目录只读？）——本次会话内仍然有效。")
    if st.session_state.pop("_rag_ui_reset_notice", False):
        st.caption("已恢复默认设置。")

    with st.expander("查看 / 导出当前设置"):
        st.code(state_to_json(state), language="json")
        st.caption("状态文件：项目根 `rag.local.json`（已在 .gitignore 中，不会提交）")
    return status


# ============================== 入口 ==============================

def main() -> None:
    st.set_page_config(page_title="RAG 观测台", page_icon="🔎", layout="wide")
    st.title("🔎 RAG 观测台")
    st.caption("看见一次提问在这条流水线里的全过程：两路召回 → 融合 → 聚合 → 重排 → "
               "预算装填 → 引用校验 → 拒答。所有数字都来自真实链路，面板不重算。")

    _init_state()
    state = _state()

    # 系统对话框选完目录后，在这里落地：此时输入框还没被创建，改它是合法的。
    # （按钮回调里不能直接改，那一轮输入框已经实例化了。）
    pending_dir = st.session_state.pop("_pending_docs_dir", None)
    if pending_dir:
        st.session_state["docs_dir_input"] = pending_dir
        _persist(dict(_state(), docs_dir=normalize_dir(pending_dir)))

    with st.sidebar:
        corpus_status = sidebar_corpus_manager()
        docs_dir = normalize_dir(st.session_state.get("docs_dir_input") or "")

        st.divider()
        st.header("检索与生成")
        use_bm25 = st.checkbox("启用 BM25 关键词通道", value=bool(state.get("use_bm25", True)),
                               help="关掉它就能看到「纯向量」的召回质量——混合检索的价值由此可见")
        use_mmr = st.checkbox("启用 MMR 多样性重排", value=bool(state.get("use_mmr", True)))
        with_generator = st.checkbox("启用生成层（离线复述）",
                                     value=bool(state.get("with_generator", True)),
                                     help="EchoGenerator：不调用任何模型，用于验证引用校验与拒答链路")
        top_k = st.slider("top_k", 1, 10, int(state.get("top_k", C.TOP_K)))
        min_sim = st.slider("min_sim（余弦下限）", 0.0, 1.0,
                            float(state.get("min_sim", 0.0)), 0.05,
                            help="伪向量要传 0：它的分数普遍偏低，用 0.35 会把结果全滤掉")
        multi_turn = st.checkbox(
            "多轮追问（把历史拼进 prompt）",
            value=bool(state.get("multi_turn", False)),
            help="4A：只把历史拼进 prompt 并扣预算，**不做查询改写**——"
                 "纯指代（如“它的默认值呢？”）仍会检索跑偏，请把问题问完整")

        # 开关变化也记住（否则每次开会话都要重设一遍）
        if (use_bm25, use_mmr, with_generator, top_k, min_sim, multi_turn) != (
                state.get("use_bm25"), state.get("use_mmr"), state.get("with_generator"),
                state.get("top_k"), state.get("min_sim"), state.get("multi_turn")):
            _persist(dict(state, use_bm25=use_bm25, use_mmr=use_mmr,
                          with_generator=with_generator, top_k=top_k, min_sim=min_sim,
                          multi_turn=multi_turn))

        st.divider()
        st.caption(f"索引签名 `{compute_index_signature(CachingEmbedder(HashEmbedder()))}`")
        st.caption("改了 config.py 的切分参数后，这个签名会变，索引会自动重建。")

    if not corpus_status.usable:
        st.error(f"语料目录不可用：{corpus_status.message}\n\n"
                 f"请在左侧修正路径后继续（面板不会因此崩溃，但检索会是空的）。")

    question = st.text_input("问题", value=DEFAULT_QUESTION)

    session = get_session(docs_dir, use_bm25, use_mmr, with_generator)
    # 多轮开关由界面传入（历史本身存在 Session 上，与 pipeline 同生命周期）
    session.multi_turn = multi_turn
    if multi_turn:
        cols = st.columns([6, 1])
        if session.turns:
            cols[0].caption(
                f"多轮历史：已记录 **{len(session.turns)}** 轮｜最近一轮："
                f"「{session.turns[-1]['question'][:40]}」"
                f"（历史会从资料预算里扣走，上限见 `config.HISTORY_MAX_TOKENS`）")
        else:
            cols[0].caption("多轮历史：还没有记录。提交一个问题后，这一轮会自动进入历史。")
        if cols[1].button("清空历史", width="stretch"):
            session.turns.clear()
            session.recorded_question = ""
            st.rerun()

    tabs = st.tabs(["① 总览", "② 检索瀑布", "③ 上下文与 prompt", "④ 生成与校验", "⑤ 评测"])
    with tabs[0]:
        tab_overview(session, docs_dir)
    result: Dict[str, Any] = {}
    with tabs[1]:
        result = tab_retrieval(session, question, top_k, min_sim)
    with tabs[2]:
        if result:
            tab_context(result)
    with tabs[3]:
        if result:
            tab_generation(session, result)
    with tabs[4]:
        tab_eval(session, docs_dir)


if __name__ == "__main__":
    main()
