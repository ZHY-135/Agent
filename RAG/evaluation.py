# -*- coding: utf-8 -*-
"""⑨ 评估层：把「检索/生成质量」变成可复现的数字。

**阶段 1（当前文件）只做三件事**：数据契约、锚点匹配、检索指标计算。
生成侧的幻觉率/误拒率指标依赖 LLM 调用，放在后续阶段（见 `evaluate_generation` 预留）。

--------------------------------------------------------------------------------
设计要点 1：为什么用「锚点」而不是 chunk id 做标注
--------------------------------------------------------------------------------
`parent_id` / `child_id` 是内容哈希派生的（`uuid5(source::idx::text)`），
文档改一个字、重新切分一次，全部 ID 都会变——金标集若存 ID，第二天就全失效。
因此标注的是「**答案在这段文本里**」，用 Anchor 三元组描述：

    source          文件名（只用 basename，换机器/换绝对路径不失效）
    heading_contains 标题路径需包含的片段（如 "常见错误"）
    text_contains   正文需包含的片段（如 "0x80070005"）

匹配在**父块**层级做：父块比子块稳定（不随滑窗参数漂移），且它才是喂给 LLM 的内容。

--------------------------------------------------------------------------------
设计要点 2：为什么按父块去重
--------------------------------------------------------------------------------
同一父块会命中多个子块（滑窗重叠 + child_index 不同）。prompt 里被引用一次，
自然不该按命中次数重复计分，否则「召回了几次」会污染「召回了几条」。
因此判定前先按 parent_id 保留最高分的一条。

--------------------------------------------------------------------------------
设计要点 3：样本量不足必须显式暴露
--------------------------------------------------------------------------------
`recall@5` 若基于 3 条样本算出 1.00，那是噪声不是成绩。
`Report.warnings` 会在样本量不足时打印告警，报告里也会写明分母（如 `12/50`）。
"""
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# pipeline 不反向依赖 evaluation（见模块依赖方向：main → pipeline → 各功能模块），
# 因此这里直接导入不会成环。运行器需要构造 pipeline，放在本模块是为了让
# 「评测逻辑」与「评测编排」在同一处，避免散落到 CLI 里难以测试。
from .config import TOP_K
from .logging_setup import get_logger

logger = get_logger(__name__)

__all__ = [
    "Anchor", "GoldenItem", "RunConfig", "MetricsConfig", "ItemResult", "Report",
    "RetrievalMetrics", "PolicyRates", "match_anchor", "judge_hits",
    "dedupe_by_parent", "coverage_at_k", "first_match_rank", "recall_at_k", "ndcg_at_k",
    "load_golden_set", "save_golden_set", "evaluate_retrieval", "evaluate_generation",
    "evaluate", "render_report", "ABLATIONS", "run_experiment", "check_anchors",
    "render_check_report",
]

# 指标口径写死在代码里而不是配置里：改了它，历史报告就不再可比，
# 必须是一次显式的、会在 code review 里被看到的改动。
METRIC_VERSION = "retrieval-v1"

DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)

# 低于这些样本量时给出告警——避免用个位数样本算出"100% 命中"当成绩
MIN_SAMPLES_ANSWERABLE = 20
MIN_SAMPLES_UNANSWERABLE = 5
MIN_SAMPLES_GENERATED = 20


# ============================== 数据契约 ==============================

@dataclass(frozen=True)
class Anchor:
    """一条「正确答案所在位置」的标注。三个条件同时满足才算命中。"""

    source: str = ""                 # 文件名（basename 比对），留空表示不限文件
    heading_contains: str = ""
    text_contains: str = ""

    def __post_init__(self) -> None:
        if not (self.source or self.heading_contains or self.text_contains):
            raise ValueError("Anchor 至少要有 source / heading_contains / text_contains 之一，"
                             "否则等于「任何块都算命中」，指标会恒为 1")

    def label(self) -> str:
        parts = [p for p in (self.source, self.heading_contains, self.text_contains) if p]
        return " | ".join(parts)


@dataclass(frozen=True)
class GoldenItem:
    """一条评测样本。"""

    qid: str
    question: str
    anchors: Tuple[Anchor, ...] = ()
    answerable: bool = True          # False 表示语料里没有答案，用来测拒答
    points: Tuple[str, ...] = ()     # 答案要点，供后续忠实度评测使用（可选）
    note: str = ""
    reviewed: bool = False           # ★ 人工核验过才计入指标：机器起草的题必须复核

    def __post_init__(self) -> None:
        if not self.qid or not self.question.strip():
            raise ValueError("GoldenItem 需要非空的 qid 与 question")
        if self.answerable and not self.anchors:
            raise ValueError(f"[{self.qid}] answerable=True 必须给出至少一个 anchor；"
                             f"若语料中确实没有答案，请显式写 answerable=false")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["anchors"] = [asdict(a) for a in self.anchors]
        data["points"] = list(self.points)
        return data

    @staticmethod
    def from_dict(raw: Dict[str, Any], where: str = "") -> "GoldenItem":
        missing = [f for f in ("qid", "question") if not raw.get(f)]
        if missing:
            raise ValueError(f"{where} 缺少必填字段 {missing}")
        anchors = tuple(Anchor(**a) for a in raw.get("anchors") or [])
        return GoldenItem(
            qid=str(raw["qid"]),
            question=str(raw["question"]),
            anchors=anchors,
            answerable=bool(raw.get("answerable", True)),
            points=tuple(raw.get("points") or ()),
            note=str(raw.get("note") or ""),
            reviewed=bool(raw.get("reviewed", False)),
        )


@dataclass(frozen=True)
class RunConfig:
    """一次评测运行的实际配置。

    三个检索开关复用 pipeline 已有的参数——评估层不发明新概念，
    否则「线上跑的」和「评测跑的」会悄悄变成两套东西。
    """

    name: str
    use_bm25: bool
    use_mmr: bool
    min_sim: Optional[float] = None      # None 表示用 pipeline 默认值

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(raw: Dict[str, Any], where: str = "") -> "RunConfig":
        if not raw.get("name"):
            raise ValueError(f"{where} 缺少 run.name")
        return RunConfig(
            name=str(raw["name"]),
            use_bm25=bool(raw.get("use_bm25", True)),
            use_mmr=bool(raw.get("use_mmr", True)),
            min_sim=raw.get("min_sim"),
        )


@dataclass(frozen=True)
class MetricsConfig:
    """指标口径配置。改动它必须同步 METRIC_VERSION，否则历史报告不可比。"""

    k_values: Tuple[int, ...] = DEFAULT_K_VALUES
    metric_version: str = METRIC_VERSION

    @property
    def k_max(self) -> int:
        return max(self.k_values)

    def to_dict(self) -> Dict[str, Any]:
        return {"k_values": list(self.k_values), "metric_version": self.metric_version}

    @staticmethod
    def from_dict(raw: Optional[Dict[str, Any]]) -> "MetricsConfig":
        if not raw:
            return MetricsConfig()
        k_values = tuple(int(k) for k in raw.get("k_values") or DEFAULT_K_VALUES)
        if not k_values or min(k_values) < 1:
            raise ValueError("metrics.k_values 必须都是 >=1 的正整数")
        return MetricsConfig(k_values=k_values,
                             metric_version=str(raw.get("metric_version", METRIC_VERSION)))


# ============================== 锚点匹配 ==============================

def match_anchor(anchor: Anchor, text: str, heading: str = "", source: str = "") -> bool:
    """判断一块文本是否落在锚点范围内。

    ⚠ `text` 是**正文**，`source` 才是**文件路径**。两个参数刻意分开：
       早期版本只用一个 `doc` 参数同时承担两者，结果「输出正文时文件名校验就失效，
       输出路径时正文校验就失效」——指标会静默变成恒真或恒假。
       （这正是 `tests/test_evaluation.py` 里两个用例抓出来的缺陷。）

    用子串包含而非正则/语义匹配：包含关系是**确定性的、可复现的**，
    评测工具一旦引入模糊匹配，指标本身就不可信了。

    >>> a = Anchor(source="d.md", heading_contains="常见错误", text_contains="0x80070005")
    >>> match_anchor(a, "遇到 error code 0x80070005 时…", "第三章 > 常见错误", r"C:\\docs\\d.md")
    True
    >>> match_anchor(a, "遇到 0x80070005", "第三章 > 配置示例", r"C:\\docs\\d.md")   # 标题不符
    False
    >>> match_anchor(a, "遇到 0x80070005", "第三章 > 常见错误", r"C:\\docs\\other.md")  # 文件不符
    False
    """
    if anchor.source and Path(anchor.source).name != Path(source).name:
        return False
    if anchor.heading_contains and anchor.heading_contains not in (heading or ""):
        return False
    if anchor.text_contains and anchor.text_contains not in (text or ""):
        return False
    return True


def dedupe_by_parent(hits: Sequence[Dict], parents: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    """按 parent_id 去重，保留得分最高的一条，并保持原有顺序。

    返回 [(parent_id, hit)]。同一父块的多个子块命中只算一条——
    prompt 里只被引用一次，按次数计分会污染「召回了几条」。

    >>> hits = [{"parent_id": "p1", "score": 0.5}, {"parent_id": "p1", "score": 0.9},
    ...         {"parent_id": "p2", "score": 0.3}]
    >>> [pid for pid, _ in dedupe_by_parent(hits, {})]
    ['p1', 'p2']
    """
    best: Dict[str, Dict] = {}
    order: List[str] = []
    for hit in hits or []:
        pid = hit.get("parent_id")
        if pid is None:
            continue
        if pid not in best:
            best[pid] = hit
            order.append(pid)
        elif float(hit.get("score") or 0.0) > float(best[pid].get("score") or 0.0):
            best[pid] = hit
    return [(pid, best[pid]) for pid in order]


def _snippet(anchor: Optional[Anchor], text: str, width: int = 60) -> str:
    """截取锚点命中处的上下文，供失败明细核对。

    ★ 只打印 heading 是不够的：同一小节常被切分成多个父块
      （实测 152 个父块只有 60 个不同标题，38 组标题被多个父块共用），
      相邻排名的 heading 因此完全相同，明细看起来像"重复行"而无法定位问题。
      必须显示**命中处的文本**才能看出差别。
    """
    if anchor is None or not anchor.text_contains:
        return ""
    pos = (text or "").find(anchor.text_contains)
    if pos < 0:
        return ""
    start = max(0, pos - 12)
    return (text[start:start + width + len(anchor.text_contains)]).replace("\n", " ")


def judge_hits(hits: Sequence[Dict], parents: Dict[str, Dict],
               anchors: Sequence[Anchor]) -> List[Dict[str, Any]]:
    """对每条检索命中判定是否落在锚点内。

    返回按 rank 排列的判定列表，每项含：
        rank         1 起，按父块去重后的排名
        parent_id
        matched      该父块命中了哪些锚点（下标列表，空表示未命中）
        best_anchor  matched 中下标最小的锚点（用于定位「最先命中的是哪个答案」）
        snippet      命中处的文本片段（未命中为空）——失败明细靠它区分同标题的相邻父块
        source / heading / score

    `best_anchor` 是必要的：一条样本可能有多个 anchor（多跳/聚合题），
    MRR 要取「最先命中的那个锚点」，而不是「命中数量」。
    """
    ranked = dedupe_by_parent(hits, parents)
    out: List[Dict[str, Any]] = []
    for rank, (pid, hit) in enumerate(ranked, start=1):
        parent = parents.get(pid) or {}
        heading = parent.get("heading") or hit.get("heading") or ""
        source = parent.get("source") or hit.get("source") or ""
        # 回表失败时退化为子块文本：仍是真实文本，不会把判定变成恒假
        text = parent.get("parent_text") or hit.get("child_text") or ""
        matched = [i for i, a in enumerate(anchors)
                   if match_anchor(a, text, heading=heading, source=source)]
        out.append({
            "rank": rank,
            "parent_id": pid,
            "matched": matched,
            "best_anchor": matched[0] if matched else None,
            "snippet": _snippet(anchors[matched[0]], text) if matched else "",
            "source": source,
            "heading": heading,
            "score": float(hit.get("score") or 0.0),
        })
    return out


def coverage_at_k(judged: Sequence[Dict[str, Any]], n_anchors: int, k: int) -> float:
    """top-k 覆盖了多少条 anchor（按父块去重后的排名）。"""
    if n_anchors <= 0:
        return 0.0
    covered = {i for row in judged[:k] for i in row["matched"]}
    return len(covered) / n_anchors


def first_match_rank(judged: Sequence[Dict[str, Any]]) -> Optional[int]:
    """第一个命中任一锚点的排名，None 表示完全未命中。"""
    return next((row["rank"] for row in judged if row["matched"]), None)


# ============================== 指标计算 ==============================

def recall_at_k(pairs: Sequence[Tuple[int, int]], n_anchors: int, k: int) -> float:
    """锚点覆盖率：top-k 覆盖了多少条 anchor。

    多锚点样本按比例给分（2 个锚点命中 1 个 = 0.5），而不是二值——
    否则「跨两段的聚合题」与「单段事实题」会被同等对待，看不出差别。

    >>> recall_at_k([(1, 0), (3, 1)], 2, 5)
    1.0
    >>> recall_at_k([(1, 0)], 2, 5)
    0.5
    >>> recall_at_k([], 2, 5)
    0.0
    """
    if n_anchors <= 0:
        return 0.0
    covered = {i for rank, i in pairs if rank <= k}
    return len(covered) / n_anchors


def ndcg_at_k(ranks: Sequence[bool], k: int) -> float:
    """二值相关性 nDCG@k：命中=1，未命中=0。

    理想 DCG 取「前 n_relevant 个位置全命中」，n_relevant = min(命中总数上限, k)。
    未命中任何内容时返回 0.0。

    >>> round(ndcg_at_k([True, False, True], 5), 4)      # DCG=1+0.5=1.5, IDCG=1+0.6309=1.6309
    0.9197
    >>> ndcg_at_k([False, False], 5)
    0.0
    """
    gains = [1.0 if r else 0.0 for r in ranks[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_rel = min(int(sum(gains)), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg else 0.0


@dataclass
class RetrievalMetrics:
    """检索侧指标。所有比率都在 answerable 样本上计算。"""

    n_items: int = 0
    n_answerable: int = 0
    n_unanswerable: int = 0
    n_expected: int = 0                 # answerable 且 reviewed 的样本数（真正的分母）
    recall: Dict[int, float] = field(default_factory=dict)
    hit: Dict[int, float] = field(default_factory=dict)
    mrr: Dict[int, float] = field(default_factory=dict)
    ndcg: Dict[int, float] = field(default_factory=dict)
    avg_used_ratio: float = 0.0         # 上下文预算利用率均值
    avg_dropped: float = 0.0            # 平均被预算挤掉的块数

    def to_dict(self) -> Dict[str, Any]:
        return {"n_items": self.n_items, "n_answerable": self.n_answerable,
                "n_unanswerable": self.n_unanswerable, "n_expected": self.n_expected,
                "recall": {str(k): v for k, v in self.recall.items()},
                "hit": {str(k): v for k, v in self.hit.items()},
                "mrr": {str(k): v for k, v in self.mrr.items()},
                "ndcg": {str(k): v for k, v in self.ndcg.items()},
                "avg_used_ratio": self.avg_used_ratio, "avg_dropped": self.avg_dropped}


@dataclass
class ItemResult:
    """单条样本的评测结果。失败明细靠它输出——这才是最有用的一部分。"""

    qid: str
    question: str
    answerable: bool
    matched_any: bool = False
    coverage: float = 0.0               # 该样本的锚点覆盖率（不截断，全深度）
    first_rank: Optional[int] = None    # 第一个命中的父块排名，None 表示完全未命中
    judged: List[Dict[str, Any]] = field(default_factory=list)
    packed_tokens: int = 0
    used_ratio: float = 0.0
    dropped: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Report:
    """一次运行的完整结果。"""

    run: RunConfig
    metrics: MetricsConfig
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    policy: Optional["PolicyRates"] = None       # 阶段 2 填充（需要生成）
    items: List[ItemResult] = field(default_factory=list)
    embedder_name: str = ""
    pseudo_vectors: bool = False
    corpus: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "metrics": self.metrics.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "policy": self.policy.to_dict() if self.policy else None,
            "embedder_name": self.embedder_name,
            "pseudo_vectors": self.pseudo_vectors,
            "corpus": self.corpus,
            "warnings": list(self.warnings),
            "items": [i.to_dict() for i in self.items],
        }


@dataclass
class PolicyRates:
    """生成侧指标（阶段 2）。字段先定下来，避免报告结构以后返工。"""

    n_generated: int = 0                 # 实际给出答案的样本数（幻觉率的分母）
    n_refused: int = 0
    hallucination_rate: float = 0.0      # 引用了不存在编号的比例
    unsupported_rate: float = 0.0        # 零合法引用的比例
    refusal_accuracy: float = 0.0        # 不可答样本中被正确拒答的比例
    miss_rate: float = 0.0               # 可答样本中被误拒的比例
    citation_coverage: float = 0.0
    refusal_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================== 单条与整体评测 ==============================

def evaluate_retrieval(
    result_hits: Sequence[Dict],
    parent_lookup: Dict[str, Dict],
    item: GoldenItem,
    k_max: int,
) -> ItemResult:
    """对一条样本的检索结果算判定，不涉及任何生成调用。"""
    judged = judge_hits(result_hits, parent_lookup, item.anchors)
    return ItemResult(
        qid=item.qid, question=item.question, answerable=item.answerable,
        matched_any=bool(judged) and any(row["matched"] for row in judged),
        coverage=coverage_at_k(judged, len(item.anchors), len(judged)),
        first_rank=first_match_rank(judged),
        judged=judged,
    )


def _aggregate(items: List[GoldenItem], results: List[ItemResult],
               metrics: MetricsConfig) -> RetrievalMetrics:
    """把单条结果聚合成报告级指标。

    ⚠ 分母只算 `answerable and reviewed` 的样本，三个条件都会转成 warnings：
       - 不可答样本本就没有 anchor，混进 recall 分母会把指标稀释成噪声；
       - 未人工核验的题是机器起草的，指标不可信，必须在分母外；
       - **检索失败的样本必须排除**：若只把它们从分子剔除、仍留在分母，
         等于把"程序报错"记成"检索没命中"——指标会被基础设施故障悄悄拉低，
         而且故障越严重指标越难看，却看不出真实原因。
    """
    pairs = list(zip(items, results))
    expected = [(i, r) for i, r in pairs if i.answerable and i.reviewed and not r.error]

    recall: Dict[int, float] = {}
    hit: Dict[int, float] = {}
    mrr: Dict[int, float] = {}
    ndcg: Dict[int, float] = {}
    for k in metrics.k_values:
        if not expected:
            recall[k] = hit[k] = mrr[k] = ndcg[k] = 0.0
            continue
        n = len(expected)
        recall[k] = sum(coverage_at_k(r.judged, len(i.anchors), k) for i, r in expected) / n
        hit[k] = sum(1 for _, r in expected
                     if r.first_rank is not None and r.first_rank <= k) / n
        mrr[k] = sum((1.0 / r.first_rank) if (r.first_rank and r.first_rank <= k) else 0.0
                     for _, r in expected) / n
        ndcg[k] = sum(ndcg_at_k([bool(row["matched"]) for row in r.judged], k)
                      for _, r in expected) / n

    usable = [r for _, r in expected if not r.error]
    return RetrievalMetrics(
        n_items=len(results),
        n_answerable=sum(1 for i in items if i.answerable),
        n_unanswerable=sum(1 for i in items if not i.answerable),
        n_expected=len(expected),
        recall=recall, hit=hit, mrr=mrr, ndcg=ndcg,
        avg_used_ratio=(sum(r.used_ratio for r in usable) / len(usable)) if usable else 0.0,
        avg_dropped=(sum(r.dropped for r in usable) / len(usable)) if usable else 0.0,
    )


def evaluate_generation(*_args: Any, **_kwargs: Any) -> PolicyRates:
    """生成侧指标（幻觉率 / 误拒率 / 拒答准确率）。**阶段 2 实现**。

    现在不做的原因：它需要真实 LLM 调用，而检索指标可以完全离线复现。
    两者混在一次运行里，会导致「跑一次消融要花钱」，回归门禁就没人跑了。
    """
    raise NotImplementedError(
        "生成侧指标属于阶段 2：需要 --provider openai（真实 LLM 调用）。"
        "当前阶段请用 --mode retrieval 评测检索质量。")


def evaluate(items: Sequence[GoldenItem], results: Sequence[ItemResult],
             metrics: MetricsConfig, run: RunConfig,
             *, embedder_name: str = "", pseudo_vectors: bool = False,
             corpus: Optional[Dict[str, Any]] = None,
             only_qids: Optional[Sequence[str]] = None) -> Report:
    """聚合为报告，并附带用于判断指标是否可信的告警。"""
    if len(items) != len(results):
        raise ValueError(f"items({len(items)}) 与 results({len(results)}) 数量不一致——"
                         f"聚合前必须一一对应，否则指标会错配到别的题上")

    mismatched = [r.qid for i, r in zip(items, results) if i.qid != r.qid]
    if mismatched:
        raise ValueError(f"items 与 results 顺序不一致（qid 对不上）：{mismatched[:5]}——"
                         f"顺序错位会让每条结果都被算到别的题的分母上")
    wrong_flag = [r.qid for i, r in zip(items, results) if i.answerable != r.answerable]
    if wrong_flag:
        raise ValueError(f"{len(wrong_flag)} 条结果的 answerable 与评测集不一致：{wrong_flag[:5]}")

    if only_qids is not None:
        wanted = set(only_qids)
        keep = [(i, r) for i, r in zip(items, results) if i.qid in wanted]
        missing = wanted - {i.qid for i, _ in keep}
        if missing:
            raise ValueError(f"only_qids 中有 {len(missing)} 个 qid 不在评测集里：{sorted(missing)[:5]}")
        items = [i for i, _ in keep]
        results = [r for _, r in keep]

    retrieval = _aggregate(list(items), list(results), metrics)

    warnings: List[str] = []
    unreviewed = sum(1 for i in items if not i.reviewed)
    if unreviewed:
        warnings.append(f"{unreviewed}/{len(items)} 条样本未标记 reviewed，已排除在指标分母之外"
                        f"（机器起草的题必须先人工核验）")
    if retrieval.n_expected < MIN_SAMPLES_ANSWERABLE:
        warnings.append(f"可答且已核验的样本仅 {retrieval.n_expected} 条，"
                        f"低于 {MIN_SAMPLES_ANSWERABLE} 条——指标波动会很大，不要当作结论")
    if retrieval.n_unanswerable < MIN_SAMPLES_UNANSWERABLE:
        warnings.append(f"不可答样本仅 {retrieval.n_unanswerable} 条，"
                        f"低于 {MIN_SAMPLES_UNANSWERABLE} 条——无法评估拒答能力")
    failed = [r.qid for r in results if r.error]
    if failed:
        warnings.append(f"{len(failed)} 条样本检索失败：{failed[:5]}")
    if pseudo_vectors:
        warnings.append("当前 embedding 是伪向量（HashEmbedder）：指标只能用于回归对比，"
                        "绝对不能作为效果结论对外引用")

    return Report(run=run, metrics=metrics, retrieval=retrieval, items=list(results),
                  embedder_name=embedder_name, pseudo_vectors=pseudo_vectors,
                  corpus=dict(corpus or {}), warnings=warnings)


# ============================== 评测运行器 ==============================

def run_experiment(items: Sequence[GoldenItem], docs: str,
                   pipeline_factory: Any, *,
                   runs: Optional[Sequence[RunConfig]] = None,
                   metrics: Optional[MetricsConfig] = None,
                   top_k: int = TOP_K,
                   min_sim: Optional[float] = None,
                   index: bool = True) -> List[Report]:
    """跑一组消融配置，产出可直接对比的 `Report` 列表。

    设计要点：

    1. **评测调用 `pipeline.query()` 而不是 `answer()`**
       检索指标必须完全离线可复现。混入 LLM 调用会让"跑一次评测要花钱"，
       于是回归门禁就没人跑了——这是评测体系失效最常见的原因。

    2. **每组配置独立建索引**
       `use_bm25` / `use_mmr` 是 pipeline 的构造参数，且 BM25 索引在 index() 后重建。
       复用同一 pipeline 会导致后面的配置带着前面配置的 BM25 状态，
       四组结果互相污染（表现为 `纯向量` 那一组也命中了大量关键词结果）。

    3. **每组配置内部复用一个 pipeline**
       一次 index()、多次 query()。否则 50 题 × 4 组要重复建 200 次索引。

    4. **先筛查询、再跑检索**
       只对计入指标或明确要看失败的样本跑检索，避免为不可答样本做无用的 embedding 调用。

    下面的示例用假 pipeline 展示编排契约（真实语料请用 `python -m RAG.main eval`）：

    >>> class FakeStore:
    ...     def list_children(self): return []
    ...     def get_child_vectors(self, ids): return {}
    ...     def get_parents(self, ids): return {}
    >>> class FakePipeline:
    ...     embedder = None
    ...     def __init__(self, **kw): self.kw = kw
    ...     def index(self, docs): return {"parents": 1, "children": 1}
    ...     def query(self, q, top_k=5, min_sim=None):
    ...         return {"hits": [{"parent_id": "p1", "score": 0.9,
    ...                           "child_text": "答案就在这句话里"}],
    ...                 "parents_used": {"p1": {"parent_id": "p1", "heading": "H",
    ...                                         "parent_text": "答案就在这句话里"}},
    ...                 "packed": {"used_tokens": 10, "budget": 100,
    ...                            "utilization": 0.1, "dropped": []}}
    >>> items = [GoldenItem(qid="q1", question="问题",
    ...                     anchors=(Anchor(text_contains="答案就在这句话里"),),
    ...                     reviewed=True)]
    >>> reports = run_experiment(items, "docs", FakePipeline,
    ...                          runs=[ABLATIONS["vector"]])
    >>> reports[0].retrieval.hit[1]
    1.0
    """
    metrics = metrics or MetricsConfig()
    runs = list(runs) if runs is not None else list(ABLATIONS.values())
    if not runs:
        raise ValueError("runs 为空：至少要有一个评测配置")

    reports: List[Report] = []
    for run in runs:
        pipeline = pipeline_factory(use_bm25=run.use_bm25, use_mmr=run.use_mmr)
        stat = pipeline.index(docs) if index else {}
        corpus = {"summary": f"{docs}（父块 {stat.get('parents', '?')} / "
                            f"子块 {stat.get('children', '?')}）"} if stat else {"summary": docs}
        logger.info("评测配置「%s」：索引就绪（父块 %s / 子块 %s）",
                    run.name, stat.get("parents", "?"), stat.get("children", "?"))

        results: List[ItemResult] = []
        for item in items:
            # 不可答样本不参与指标，但失败明细里保留它的判定，便于人工复核
            try:
                outcome = pipeline.query(
                    item.question, top_k=top_k,
                    min_sim=run.min_sim if run.min_sim is not None else min_sim)
            except Exception as error:                     # noqa: BLE001
                # 单条失败不能中断整轮评测：记录错误，指标分母会把该条排除并告警
                logger.warning("样本 %s 检索失败，已从指标分母排除：%s: %s",
                               item.qid, type(error).__name__, error)
                results.append(ItemResult(
                    qid=item.qid, question=item.question, answerable=item.answerable,
                    error=f"{type(error).__name__}: {error}"))
                continue

            single = evaluate_retrieval(outcome["hits"], outcome.get("parents_used") or {},
                                        item, metrics.k_max)
            packed = outcome.get("packed") or {}
            single.packed_tokens = int(packed.get("used_tokens") or 0)
            single.used_ratio = float(packed.get("utilization") or 0.0)
            single.dropped = len(packed.get("dropped") or [])
            results.append(single)
            logger.debug("样本 %s 首个命中排名=%s，覆盖率=%.2f",
                         item.qid, single.first_rank, single.coverage)

        report = evaluate(
            list(items), results, metrics, run,
            embedder_name=_embedder_name(pipeline),
            pseudo_vectors=bool(getattr(pipeline.embedder, "pseudo_vectors", False)),
            corpus=corpus)
        logger.info("配置「%s」完成：recall@5=%.3f，hit@1=%.3f",
                    run.name, report.retrieval.recall.get(5, 0.0),
                    report.retrieval.hit.get(1, 0.0))
        reports.append(report)
    return reports


def _embedder_name(pipeline: Any) -> str:
    """取嵌入器名字，尽量把装饰器剥掉只留内层实现（CachingEmbedder 之类不提供信息）。"""
    embedder = getattr(pipeline, "embedder", None)
    inner = getattr(embedder, "inner", embedder)
    return type(inner).__name__ if inner is not None else ""


# ============================== 评测集读写 ==============================
def load_golden_set(path: str) -> List[GoldenItem]:
    """从 JSONL 读取评测集。每行一个样本，`#` 开头的行与空行忽略。

    校验是刻意啰嗦的：评测集的字段错误若被静默接受，
    会变成「指标看起来正常但实际在测错的东西」——那比直接报错危险得多。
    """
    items: List[GoldenItem] = []
    seen_qid: Dict[str, int] = {}
    seen_question: Dict[str, int] = {}
    seen_anchor: Dict[Tuple[str, str], int] = {}
    text = Path(path).read_text(encoding="utf-8")

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{lineno} 不是合法 JSON：{error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{lineno} 每行必须是一个 JSON 对象")

        item = GoldenItem.from_dict(raw, f"{path}:{lineno}")
        if item.qid in seen_qid:
            raise ValueError(f"{path}:{lineno} qid 重复：{item.qid}（首次出现在第 {seen_qid[item.qid]} 行）")
        seen_qid[item.qid] = lineno

        key = item.question.strip()
        if key in seen_question:
            raise ValueError(f"{path}:{lineno} 问题与第 {seen_question[key]} 行重复：{item.question!r}")
        seen_question[key] = lineno

        for a in item.anchors:
            # 唯一性以「锚点 + 抽取规则」为键：
            #   · 同一规则重复出题 = 复制粘贴标注后忘了改，必须拦下；
            #   · 但**不同规则共用一行 anchor 是合法的**——一行 Markdown 表格里
            #     可能同时抽到多个配置项（如 `TOP_K` 与 `CANDIDATE_MUL`），
            #     它们的锚点自然相同。若一律拒绝，起草器就无法产出这类题。
            rule = item.note.split("；")[0].strip()
            # 用不同变量名（anchor_key）而非复用上面的 key：
            # 上面的 key 是 str（问题文本），这里需要 tuple。复用同名变量会让
            # 类型推断出错、也让读者误以为两者是同一个东西。
            anchor_key = (a.label(), rule)
            if anchor_key in seen_anchor:
                raise ValueError(
                    f"{path}:{lineno} 锚点与第 {seen_anchor[anchor_key]} 行重复：{a.label()}"
                    f"（规则 {rule or '未标注'}）——同规则下同一段文本被两题共用，"
                    f"通常说明标注是从别处复制后忘了改")
            seen_anchor[anchor_key] = lineno
        items.append(item)
    return items


def save_golden_set(items: Sequence[GoldenItem], path: str) -> None:
    """写出 JSONL（供起草器与人工核验后回写）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(i.to_dict(), ensure_ascii=False) for i in items]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def check_anchors(items: Sequence[GoldenItem],
                  children: Sequence[Dict]) -> List[Dict[str, Any]]:
    """自助检：在已建好的索引里检查每个锚点是否真能匹配到东西。

    标注最常见的错误不是"标错位置"，而是**拼错字符**（少一个空格、全角半角混用、
    引用了别的文件名）。这类错误不会报错，只会让那条样本永远算"未命中"，
    指标被静默拉低——所以必须有一道能提前暴露它的检查。

    返回未匹配的锚点清单，每项含 qid / 锚点内容 / 实际候选（供人工核对到底该写什么）。
    """
    # 预建索引：文件名 → [(heading, text)]，只在同一文件内比对，避免跨文件误报
    by_source: Dict[str, List[Tuple[str, str]]] = {}
    for row in children or []:
        source = row.get("source") or ""
        if not source:
            continue
        by_source.setdefault(Path(source).name, []).append(
            (row.get("heading") or "", row.get("parent_text") or row.get("child_text") or ""))

    unmatched: List[Dict[str, Any]] = []
    for item in items:
        for anchor in item.anchors:
            if anchor.source:
                name = Path(anchor.source).name
                pool = by_source.get(name)
                if not pool:
                    unmatched.append({"qid": item.qid, "anchor": anchor.label(),
                                      "reason": f"索引里没有文件 {name}",
                                      "hint": f"索引中共 {len(by_source)} 个文件"})
                    continue
                candidates = [(name, heading, text) for heading, text in pool]
            else:
                candidates = [(src, heading, text)
                              for src, pairs in by_source.items() for heading, text in pairs]

            if not any(match_anchor(anchor, text, heading=heading, source=src)
                       for src, heading, text in candidates):
                hint = ""
                if anchor.text_contains:
                    # 找出含一半关键词的候选，帮助判断是拼写问题还是标错了位置
                    needle = anchor.text_contains[:max(1, len(anchor.text_contains) // 2)]
                    near = [f"{src} :: {(heading or '')[:30]}"
                            for src, heading, text in candidates if needle in text]
                    hint = ("附近候选：" + "；".join(near[:3])) if near else "没有任何块包含该片段的前半部分"
                unmatched.append({"qid": item.qid, "anchor": anchor.label(),
                                  "reason": "索引中找不到匹配的块", "hint": hint})
    return unmatched


def render_check_report(items: Sequence[GoldenItem], unmatched: Sequence[Dict[str, Any]],
                        corpus: Optional[Dict[str, Any]] = None) -> str:
    """渲染锚点自检报告。"""
    n_anchors = sum(len(i.anchors) for i in items)
    lines = ["# 评测集锚点自检", "",
             f"- 样本 {len(items)} 条（可答 {sum(1 for i in items if i.answerable)}"
             f" / 不可答 {sum(1 for i in items if not i.answerable)}）"
             f"，已核验 {sum(1 for i in items if i.reviewed)} 条",
             f"- 锚点共 {n_anchors} 个，其中 **{len(unmatched)} 个在索引里找不到匹配**",
             f"- 语料：{(corpus or {}).get('summary') or '未记录'}",
             "- 依赖：本检查只看文字（不调用 embedding / LLM），结果与检索配置无关，可完全复现", ""]
    if not unmatched:
        lines += ["✅ 全部锚点都能在索引中匹配到内容。", ""]
        return "\n".join(lines)

    lines += ["## 未匹配的锚点（多为拼写错误或文件写错）", ""]
    for row in unmatched:
        lines.append(f"- `{row['qid']}` — `{row['anchor']}`")
        lines.append(f"  - 原因：{row['reason']}")
        if row.get("hint"):
            lines.append(f"  - {row['hint']}")
    lines += ["", "> 请逐条核对：锚点写错不会报错，只会让该样本永远算「未命中」，指标被静默拉低。"]
    return "\n".join(lines)


# ============================== 四组消融配置 ==============================

ABLATIONS: Dict[str, RunConfig] = {
    "vector":    RunConfig(name="纯向量", use_bm25=False, use_mmr=False),
    "bm25":      RunConfig(name="+BM25 混合", use_bm25=True, use_mmr=False),
    "mmr":       RunConfig(name="+BM25+MMR", use_bm25=True, use_mmr=True),
    "full":      RunConfig(name="全开（含重排与预算）", use_bm25=True, use_mmr=True),
}


# ============================== 报告渲染 ==============================

def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_report(reports: Sequence[Report], fmt: str = "markdown",
                  top_failures: int = 10) -> str:
    """渲染消融对比表 + 失败明细 + 元信息。

    元信息（embedding 模型、语料规模、指标口径版本）不是装饰：
    缺少它们的指标无法复现，也就无法作为对外引用。
    """
    if fmt not in {"markdown", "json"}:
        raise ValueError("fmt 只能是 markdown 或 json")
    if fmt == "json":
        return json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=2)
    if not reports:
        return "（没有任何评测结果）\n"

    first = reports[0]
    ks = sorted(first.metrics.k_values)
    lines: List[str] = ["# RAG 检索质量评测报告", ""]

    if first.pseudo_vectors:
        lines += ["> ⚠ **伪向量（HashEmbedder）· 仅供回归对比，不可作为效果结论**", ""]

    lines += ["## 元信息", "",
              f"- 指标口径：`{first.metrics.metric_version}`，k = {ks}",
              f"- embedding：`{first.embedder_name or '未记录'}`"
              f"{'（伪向量）' if first.pseudo_vectors else ''}",
              f"- 语料：{first.corpus.get('summary') or '未记录'}",
              f"- 样本：{first.retrieval.n_items} 条"
              f"（可答 {first.retrieval.n_answerable} / 不可答 {first.retrieval.n_unanswerable}）"
              f"，计入指标 {first.retrieval.n_expected} 条", ""]

    header = ["配置"] + [f"recall@{k}" for k in ks] + ["hit@1"] + \
             [f"MRR@{ks[-1]}"] + [f"nDCG@{ks[-1]}"] + ["预算利用率", "平均丢弃"]
    lines += ["## 消融对比", "", "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)]
    for report in reports:
        m = report.retrieval
        row = [report.run.name]
        row += [_fmt(m.recall.get(k)) for k in ks]
        row += [_fmt(m.hit.get(1))]
        row += [_fmt(m.mrr.get(ks[-1])), _fmt(m.ndcg.get(ks[-1]))]
        row += [f"{m.avg_used_ratio * 100:.0f}%", f"{m.avg_dropped:.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if first.policy is not None:
        p = first.policy
        lines += ["## 生成与拒答", "",
                  "| 给出答案 | 拒答 | 幻觉率 | 无据率 | 拒答准确率 | 误拒率 | 引用覆盖率 |",
                  "|---|---|---|---|---|---|---|",
                  "| {n} | {refused} | {hal} | {uns} | {racc} | {miss} | {cov} |".format(
                      n=p.n_generated, refused=p.n_refused,
                      hal=_fmt(p.hallucination_rate),
                      uns=_fmt(p.unsupported_rate),
                      racc=_fmt(p.refusal_accuracy),
                      miss=_fmt(p.miss_rate),
                      cov=_fmt(p.citation_coverage)),
                  ""]
        if p.refusal_reasons:
            lines += ["拒答原因分布：" +
                      "，".join(f"`{k}`×{v}" for k, v in sorted(p.refusal_reasons.items())), ""]

    all_warnings = sorted({w for r in reports for w in r.warnings})
    if all_warnings:
        lines += ["## ⚠ 指标可信度告警", ""] + [f"- {w}" for w in all_warnings] + [""]

    lines += ["## 失败明细（未命中 / 未排在第一）", ""]
    shown = 0
    for report in reports:
        for item in report.items:
            if shown >= top_failures:
                break
            if not item.answerable:
                continue
            # 只有「首个命中就排第一」才算完美命中，其余都值得看一眼
            if item.first_rank == 1:
                continue
            lines.append(f"### [{report.run.name}] {item.qid}：{item.question}")
            lines.append("")
            lines.append(f"- 首个命中排名：{item.first_rank if item.first_rank else '**未命中**'}"
                         f"，锚点覆盖率 {item.coverage * 100:.0f}%")
            if item.error:
                lines.append(f"- 检索异常：`{item.error}`")
            for judged_row in item.judged[:5]:
                mark = "✅" if judged_row["matched"] else "❌"
                # 名字取 judged_row 而非 row：本函数上方已用 row 承载"表格行"（list[str]），
                # 同名复用会让静态检查推断出错乱类型（也会让人误读）。
                row_where_text = (judged_row["heading"] or judged_row["source"]
                                  or "").split(" > ")[-1][:40]
                lines.append(f"  - {mark} #{judged_row['rank']} "
                             f"score={judged_row['score']:.4f} {row_where_text}")
                if judged_row.get("snippet"):
                    lines.append(f"      命中片段：…{judged_row['snippet']}…")
            lines.append("")
            shown += 1

    return "\n".join(lines)
