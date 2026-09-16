# -*- coding: utf-8 -*-
"""⑧ 生成层：LLM 调用 + [n] 引用回填校验 + 拒答策略。

补齐的是 RAG 面试必问、而此前完全无法回答的一环：**如何抑制幻觉**。

本模块给出三层防线，按「从便宜到昂贵」排列：

    第一层（生成前·零成本）拒答门禁 refuse_reason()
        packed 为空 → 直接拒答，不浪费一次 LLM 调用；
        向量支持度不足 → 直接拒答，不给模型「硬编」的机会。

    第二层（生成中·约束）render_prompt(..., system=ANSWER_SYSTEM_PROMPT, strict=True)
        用系统提示强制「只依据资料作答 + 每个事实性陈述标注 [n] + 资料不足时明说」。

    第三层（生成后·可检测）validate_citations()
        解析答案里的 [n]，回填到 packed["blocks"] 做校验：
          · invalid_refs —— 引用了不存在的编号（凭空编号，最强幻觉信号）
          · unsupported  —— 整段答案一个合法引用都没有（无据结论）
        校验结果写入 answer["validation"]，可据此告警、重试或降级为拒答。

⚠ 诚实说明：这三层只能**降低**幻觉概率并让问题**可被发现**，不能根治。
   validate_citations 是启发式检查（编号存在性 + 引用覆盖率），
   它检不出「引用了正确编号但曲解了原文」这类幻觉——那需要忠实度评测（见 README P1-4）。
"""
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .config import (
    ANSWER_SYSTEM_PROMPT,
    GENERATE_BACKOFF,
    GENERATE_RETRY,
    MIN_ANSWER_CITATIONS,
    REFUSAL_MESSAGES,
    REFUSE_MIN_VEC_SCORE,
    SCORE_COSINE,
    SCORE_KIND_KEY,
    USE_CITATION_CONSTRAINT,
)
from .logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_REFUSAL_MESSAGES: Dict[str, str] = dict(REFUSAL_MESSAGES)

# 匹配 [1] / [12] / [1,2] / [1-3] 三种写法，[] 内只允许数字与 , - 空格
_CITE_PATTERN = re.compile(r"\[([0-9][0-9,\- ]*)\]")

# 内层出现的引用标记：用于识别 [1[2]] 这类嵌套畸形写法，避免误报为「引用了编号 1」
_INNER_CITE_PATTERN = re.compile(r"\[[0-9][0-9,\- ]*\[")


# ============================== 引用解析 ==============================

def _expand_group(raw: str) -> List[int]:
    """把 `1,2` / `1-3` / `1, 3-5` 展开成 [1,2] / [1,2,3] / [1,3,4,5]。"""
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                if lo <= hi and hi - lo <= 50:        # 上限防止 [1-99999] 构造超大列表
                    out.extend(range(lo, hi + 1))
                elif lo_s.isdigit():                  # 反向或跨度过大：退化为只取左值
                    out.append(lo)
        elif part.isdigit():
            out.append(int(part))
    return out


def extract_citations(text: str) -> List[int]:
    """提取答案中的引用编号，按出现顺序去重返回。"""
    found: List[int] = []
    for m in _CITE_PATTERN.finditer(text or ""):
        if _INNER_CITE_PATTERN.search(m.group(0)):    # 畸形嵌套 [1[2]]：整体跳过
            continue
        for ref in _expand_group(m.group(1)):
            if ref not in found:
                found.append(ref)
    return found


def validate_citations(answer_text: str, packed: Dict) -> Dict[str, Any]:
    """把答案里的 [n] 回填到 packed["blocks"] 校验。

    返回（同时写入 answer["validation"] 的字段）：
        refs          答案引用的编号，按出现顺序
        valid_refs    refs 中真实存在于 packed 的编号（可直接用于溯源）
        invalid_refs  引用了不存在的编号 ← 凭空编号，最强幻觉信号
        unused_refs   装填了但没被引用的编号 ← 召回冗余，可用于调小 top_k
        unsupported   是否存在无据结论（合法引用数 < MIN_ANSWER_CITATIONS）
        coverage      命中率 = 合法引用去重数 / 装填块数
        ok            综合结论：无非法编号且非无据
    """
    blocks = (packed or {}).get("blocks") or []
    available = {int(b["ref"]): b for b in blocks if b.get("ref") is not None}
    refs = extract_citations(answer_text)
    valid = [r for r in refs if r in available]
    invalid = [r for r in refs if r not in available]
    used = set(valid)
    unsupported = len(used) < MIN_ANSWER_CITATIONS
    return {
        "refs": refs,
        "valid_refs": valid,
        "invalid_refs": invalid,
        "unused_refs": sorted(set(available) - used),
        "unsupported": unsupported,
        "coverage": (len(used) / len(available)) if available else 0.0,
        "ok": not invalid and not unsupported,
    }


def citation_details(refs: Sequence[int], packed: Dict) -> List[Dict[str, Any]]:
    """把引用编号解析成可溯源的出处清单（编号 → 文件 / 标题）。

    用 ref 映射而非 `blocks[ref-1]` 下标取值——后者隐含「编号必然从 1 连续递增」的假设，
    一旦上游改了编号规则就会静默取错块或 IndexError。
    """
    by_ref = {int(b["ref"]): b for b in ((packed or {}).get("blocks") or [])
              if b.get("ref") is not None}
    out: List[Dict[str, Any]] = []
    for ref in refs:
        block = by_ref.get(int(ref))
        if block is None:
            continue
        out.append({"ref": int(ref), "source": block.get("source"),
                    "heading": block.get("heading"), "kind": block.get("kind")})
    return out


def empty_validation() -> Dict[str, Any]:
    """未生成任何答案时的校验结果（拒答 / 空答案共用同一形状）。"""
    return {"refs": [], "valid_refs": [], "invalid_refs": [], "unused_refs": [],
            "unsupported": True, "coverage": 0.0, "ok": False}


# ============================== 拒答门禁 ==============================

def support_score(hits: Sequence[Dict], vectors: Optional[Dict[str, List[float]]] = None,
                  query_vec: Optional[Sequence[float]] = None) -> float:
    """检索结果的「向量支持度」= 候选子块的最高**原始余弦相似度**。

    ★ 为什么不能直接用 h["score"] 做门槛？
      混合检索走的是 RRF 融合，融合分 ≈ 1/(k+rank) ≈ 0.016 量级，
      与余弦相似度（[0,1]）差两个数量级——拿它和 REFUSE_MIN_VEC_SCORE 比，
      等于所有查询都会被判成「支持度不足」。这与 MMR 那次量纲事故是同一类错误。
      （`docs/ENGINEERING_REVIEW.md` §8.3）

    取分顺序（任一可用即可，全不可用时返回 0.0）：
      1. hit 上已缓存的 "vec_score"（pgvector 的 `1 - (embedding <=> q)` 直接给出）；
      2. 用 vectors + query_vec 现场算余弦（MemoryStore 场景，向量已在手边）。

    ★ **量纲防线**：每条第 1 步之前会检查 hit 上自带的量纲标记。
      传进来的若是 BM25 分或 RRF 融合分（`config.SCORE_BM25` / `SCORE_RRF`），
      本函数**直接抛 ValueError**，而不是算出一个"看起来很小"的数字。
      理由：这种误用会让所有查询都触发 weak_support 拒答，而排查者看到的
      现象是"检索坏了"——把静默的错误结论变成当场失败，能省掉几小时排查。
    """
    # 先整体检查量纲，再取值：这样即使 hits 为空也不会漏掉调用方的用法错误
    for h in hits or []:
        kind = h.get(SCORE_KIND_KEY)
        if kind is not None and kind != SCORE_COSINE:
            raise ValueError(
                f"support_score 只接受原始余弦相似度，收到 {kind!r} 量纲的分数。\n"
                f"    正确用法：传向量通道的原始结果（pipeline.retrieve() 的 vec_hits）。\n"
                f"    错误用法：传融合后的 hits——RRF 融合分 ≈0.016 量级，"
                f"与阈值 {REFUSE_MIN_VEC_SCORE} 差两个数量级，会导致所有查询被拒答。\n"
                f"    参见 docs/ENGINEERING_REVIEW.md §8.3（MMR 量纲事故）。")

    best = 0.0
    for h in hits or []:
        score = h.get("vec_score")
        if score is None and vectors and query_vec:
            # child_id 缺失时 .get(None) 取不到向量 → 返回 None → 余弦按 0 处理。
            # 显式传 str 是为了让静态检查能确认键类型（hit 是裸 dict，键可能是任何类型）。
            cid = h.get("child_id")
            score = cosine_similarity(vectors.get(str(cid)) if cid is not None else None,
                                      query_vec)
        if score is not None:
            best = max(best, float(score))
    return best


def refuse_reason(packed: Dict, vec_score: Optional[float] = None) -> Optional[str]:
    """生成前的拒答门禁：返回拒答原因码，None 表示可以正常生成。

    放在调用 LLM **之前**——省一次调用，也省掉一次「模型硬编」的机会。
    """
    blocks = (packed or {}).get("blocks") or []
    if not blocks:
        return "no_context"
    if vec_score is not None and vec_score < REFUSE_MIN_VEC_SCORE:
        return "weak_support"
    return None


# ============================== 生成器 ==============================

class Generator:
    """接口：所有生成器都实现 generate(prompt)。

    刻意只收 prompt 字符串——上下文装填、引用约束、拒答都是编排层的事，
    换 LLM 供应商时不需要重新实现这些策略。
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """流式输出。默认实现退化为一次性返回，子类可覆盖。"""
        yield self.generate(prompt)

    def answer(self, packed: Dict, prompt: str, question: str = "",
               vec_score: Optional[float] = None,
               strict_citations: bool = USE_CITATION_CONSTRAINT) -> Dict[str, Any]:
        """★ 生成 + 回填校验 + 拒答，统一出口。

        返回：{"text", "refused", "refusal_reason", "validation", "citations", "answer_tokens"}
        """
        # ---- 第一层：生成前门禁 ----
        reason = refuse_reason(packed, vec_score)
        if reason:
            # 拒答是要被观测的事件：原因码分布能直接反映语料覆盖与阈值是否合适
            logger.info("拒答（生成前）：reason=%s，支持度=%s，装填块数=%d",
                        reason, vec_score, len((packed or {}).get("blocks") or []))
            return self._refusal(reason, packed)

        # ---- 第二层：调用 LLM（已由 render_prompt 注入引用约束）----
        try:
            text = self.generate(prompt) or ""
        except Exception as error:                         # noqa: BLE001
            # 生成失败不退化成「编一个答案」，而是明确拒答——失败要可见
            logger.error("生成失败：%s: %s", type(error).__name__, error)
            return self._refusal("generator_error", packed)

        text = text.strip()
        if not text:
            return self._refusal("empty_answer", packed)

        # ---- 第三层：引用回填校验 ----
        validation = validate_citations(text, packed)

        if strict_citations:
            # 引用了不存在的编号 → 凭空编号，最强幻觉信号，直接拒答。
            # 注意这不是「格式不美」的问题：编号不存在意味着答案里的那句话
            # 没有任何资料支撑，只是看起来像有出处——比不标引用更危险。
            if validation["invalid_refs"]:
                logger.warning("检出凭空引用编号 %s，已拒答（幻觉信号）",
                               validation["invalid_refs"])
                result = self._refusal("invalid_citation", packed)
                result["raw_text"] = text
                result["validation"] = validation
                return result
            # 整段答案一个合法引用都没有 → 无据结论，同样降级为拒答。
            # 「降级」而非「丢弃」：原文留在 raw_text 里，便于判断是模型问题还是解析问题。
            if validation["unsupported"]:
                logger.warning("答案无合法引用（无据结论），已拒答")
                result = self._refusal("not_supported", packed)
                result["raw_text"] = text
                result["validation"] = validation
                return result

        return {
            "text": text,
            "refused": False,
            "refusal_reason": None,
            "validation": validation,
            "citations": citation_details(validation["valid_refs"], packed),
            "answer_tokens": estimate_tokens(text),
        }

    def _refusal(self, reason: str, packed: Dict) -> Dict[str, Any]:
        """构造拒答结果。拒答也是一种正常输出，结构必须与成功路径一致。"""
        return {
            "text": DEFAULT_REFUSAL_MESSAGES.get(reason, DEFAULT_REFUSAL_MESSAGES["no_context"]),
            "refused": True,
            "refusal_reason": reason,
            "validation": empty_validation(),
            "citations": [],
            "answer_tokens": 0,
        }


class EchoGenerator(Generator):
    """零依赖生成器：把装填好的资料按引用编号复述出来。

    用途：离线跑通生成层全链路 / 单元测试拒答与引用校验。
    它**通不过 MIN_ANSWER_CITATIONS 的语义考验**，但会老老实实标注 [n]，
    因此适合验证「引用解析 → 回填 → 覆盖率统计」这条链路本身是否正确。
    """

    def __init__(self, max_blocks: int = 3):
        self.max_blocks = max_blocks

    def generate(self, prompt: str) -> str:
        blocks = _parse_blocks_from_prompt(prompt)
        if not blocks:
            return "根据现有资料无法回答：未检索到相关资料。"
        lines = ["根据检索到的资料："]
        for ref, heading, text in blocks[:self.max_blocks]:
            label = f"（{heading}）" if heading else ""
            lines.append(f"- {text[:120]}{label} [{ref}]")
        return "\n".join(lines)


class OpenAIGenerator(Generator):
    """生产适配器：换掉 client 即可，生成策略（拒答 / 引用校验）完全复用。"""

    def __init__(self, client, model: str = "gpt-4o-mini",
                 temperature: float = 0.0, max_tokens: int = 800,
                 retry: int = GENERATE_RETRY, backoff: float = GENERATE_BACKOFF):
        # temperature 默认 0：RAG 要的是复述资料，不是创作
        self.client, self.model = client, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.retry, self.backoff = retry, backoff

    def generate(self, prompt: str) -> str:
        messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        last_error: Optional[Exception] = None
        attempts = max(1, self.retry)
        for attempt in range(attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens)
                # 用 getattr 而非链式下标：SDK 版本差异下 choices 可能为空
                return (getattr(resp.choices[0].message, "content", None) or "").strip()
            except Exception as error:                    # noqa: BLE001
                last_error = error
                if attempt == attempts - 1:
                    raise
                time.sleep(self.backoff * (2 ** attempt))
        raise last_error if last_error else RuntimeError("生成失败")

    def generate_stream(self, prompt: str) -> Iterator[str]:
        messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens, stream=True)
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(getattr(choices[0], "delta", None), "content", None)
            if delta:
                yield delta


# ============================== 工具函数 ==============================

def cosine_similarity(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """余弦相似度。任一为空或零向量返回 0.0（而非抛错）——支持度是软信号。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def estimate_tokens(text: str) -> int:
    """与 context.estimate_tokens 保持一致的粗估，避免两处口径漂移。"""
    from .context import estimate_tokens as _estimate
    return _estimate(text)


def _parse_blocks_from_prompt(prompt: str) -> List[Tuple[int, str, str]]:
    """从渲染好的 prompt 里解析出 [(ref, heading, text)]。

    仅 EchoGenerator 使用：它没有 packed 结构，只能从 prompt 反解。
    """
    out: List[Tuple[int, str, str]] = []
    if "===== 参考资料 =====" not in prompt:
        return out
    body = prompt.split("===== 参考资料 =====", 1)[1].split("===== 问题 =====", 1)[0]
    for chunk in re.split(r"\n(?=\[\d+\] 来源：)", body.strip()):
        chunk = chunk.strip()
        lines = chunk.split("\n")
        if not lines:
            continue
        head = re.match(r"\[(\d+)\]\s*来源：(.*)", lines[0])
        if not head:
            continue
        label = head.group(2)
        # 渲染格式为「来源：<path>[ / <heading>]」，取第一个分隔符之后的标题部分
        heading = label.split(" / ", 1)[1].strip() if " / " in label else ""
        text = "\n".join(lines[1:]).strip()
        out.append((int(head.group(1)), heading, text))
    return out
