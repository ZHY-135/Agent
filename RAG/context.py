# -*- coding: utf-8 -*-
"""⑦ 上下文组装：token 预算 + 去重 + 引用编号。

--------------------------------------------------------------------------------
这一层要解决的问题：把"检索到的资料"塞进"有限窗口"里
--------------------------------------------------------------------------------
LLM 的上下文窗口是硬限制。超限的后果不是"效果变差"，而是 **API 直接报错**。
真实场景（top_k=10 + 3000 token 的历史对话）：
    无预算控制 → 14140 token，超 6000 预算 8140 → 请求失败
    贪心装填   → 装 3 块用 5719 token（利用率 95%）→ 请求成功且信息量足够

--------------------------------------------------------------------------------
为什么用"贪心装填"而不是"按比例缩小每块"
--------------------------------------------------------------------------------
另一种常见思路是"把每块都截断一下、凑够预算"。但那样每块都不完整，
LLM 看到的是 5 段被腰斩的资料——**每段都读不通**。
贪心装填的思路是反过来的：
    按相关性从高到低，能整块放下就放；放不下就**跳过它，继续试后面的小块**。
于是结果要么是"完整的块"，要么不放，不会出现半截内容。
装不下的整块不是直接丢：先尝试降级为"命中的子块片段"（更短，但仍是完整片段）。

--------------------------------------------------------------------------------
Reader's guide
--------------------------------------------------------------------------------
    estimate_tokens()  —— token 数粗估（本模块所有预算计算的基础）
    pack_context()     —— 核心：按预算把命中装填成 blocks，产出引用编号
    render_prompt()    —— 把装填结果渲染成最终发给 LLM 的字符串
"""
from typing import Dict, List, Optional

from .config import (
    ANSWER_SYSTEM_PROMPT,
    CHARS_PER_TOKEN,
    CONTEXT_BUDGET,
    SCORE_KIND_KEY,
    USE_CITATION_CONSTRAINT,
)


def estimate_tokens(text: str) -> int:
    """粗估 token 数。生产建议用 tiktoken 精确计算。

    为什么不直接调 tiktoken：
        · 它是额外依赖，而本项目的核心路径坚持零依赖；
        · 预算是"留有余量"的约束，粗估 + 保守取值通常够用。
    粗估系数 CHARS_PER_TOKEN=1.5 是按中文标定的（中文约 1.5 字符/token）。
    误差提示：代码块与纯英文段落的实际 token 密度更高，粗估会**低估**，
    因此预算本身留了余量；高精度需求见改进方案 P1-4。
    """
    # max(1, …)：空文本也要算 1。返回 0 会让"固定开销"的累加出现黑洞——
    # 例如 system_tokens + history + query 里若 query 为空就算作 0，
    # 后续的预算比较会比实际乐观。
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def pack_context(hits: List[Dict], parents: Dict[str, Dict],
                 query: str = "", history_tokens: int = 0,
                 budget: int = CONTEXT_BUDGET,
                 system_tokens: int = 400) -> Dict:
    """按 score 贪心装填，装不下的降级为子块片段。

    参数的理解方式（把预算想成"一个固定的钱包"）：
        budget          总收入
        system_tokens   系统提示词的固定支出（每次都要付）
        history_tokens  多轮历史的固定支出
        query           当前问题的固定支出
        remaining       剩下的钱，用来"买"资料块

    返回：
        blocks       已装填并编号的块（可直接拼进 prompt）
        dropped      因预算不足被放弃的命中（用于日志/告警，不要静默丢弃）
        used_tokens  实际占用
        utilization  利用率（长期偏低说明预算过大；长期 100% 且 dropped 多说明预算过小）
    """
    # 先把三笔"固定支出"扣掉，剩下的才是可用于资料的预算。
    # 顺序很重要：如果先装资料再扣固定支出，就会出现"资料装满、系统提示词放不下"，
    # 而系统提示词是**必须**的（它承载着防幻觉约束），不能让它被挤掉。
    fixed = system_tokens + history_tokens + estimate_tokens(query)
    remaining = budget - fixed

    packed, dropped = [], []
    # hits 已按 score 降序 —— 贪心的前提是先看最相关的
    for h in hits:
        # 回表取父块全文：检索命中的是子块，但给 LLM 看的是父块（上下文更完整）
        p = parents.get(h["parent_id"])
        if not p:
            # 取不到父块说明存储层与检索结果不一致（如刚被删除）。
            # 跳过即可，不要抛出——一条异常不该让整个问答失败。
            continue
        full_tokens = estimate_tokens(p["parent_text"])
        if full_tokens <= remaining:
            # ---- 情况 A：整块放得下 → 放整块（信息最完整） ----
            packed.append({
                "kind": "parent", "parent_id": p["parent_id"],
                "text": p["parent_text"], "tokens": full_tokens,
                "source": p.get("source"), "heading": p.get("heading"),
                "score": h["score"],
                # 分量纲随块一起带上：可视化面板要在预算表里标出"这个分数是
                # 余弦还是 RRF 融合分"——两者差两个数量级，不标就会被误读。
                SCORE_KIND_KEY: h.get(SCORE_KIND_KEY),
            })
            remaining -= full_tokens
        else:
            # ---- 情况 B：整块放不下 → 降级为命中的子块片段 ----
            # 为什么不直接丢弃：这个父块是本轮最相关的资料之一，
            # 它的"命中片段"（子块）虽然短，但恰恰是与问题最贴合的那部分，
            # 放进去远比空着或换成次相关的大块更有价值。
            frag = h.get("child_text", "")
            # ⚠ 空片段必须直接丢弃：estimate_tokens("") 会返回 1，
            #   若不拦，一个空的 child_fragment 也能"装进去"，
            #   于是它占掉一个 block 名额、把真正的 dropped 记录挤掉，
            #   结果是 prompt 里混进空块、而 dropped 里什么都看不到。
            t = estimate_tokens(frag) if frag.strip() else 0
            if frag.strip() and t <= remaining:
                packed.append({
                    "kind": "child_fragment", "parent_id": p["parent_id"],
                    "text": frag, "tokens": t, "source": p.get("source"),
                    "heading": p.get("heading"), "score": h["score"],
                    SCORE_KIND_KEY: h.get(SCORE_KIND_KEY),
                })
                remaining -= t
            else:
                # ---- 情况 C：连片段都放不下 → 记录到 dropped ----
                # 必须留痕：否则"为什么这条没进 prompt"永远查不出来。
                # 上层可据此告警（例如 dropped 长期 > 0 说明预算或父块尺寸需要调整）。
                dropped.append(h)

    # 引用编号：让 LLM 能标注 [1][2]，便于溯源与人工核查。
    # 编号必须在**装填完成之后**才分配——因为只有这时才知道最终有哪几块、
    # 各是第几号。编号同时也是生成层做"[n] 引用回填校验"的依据。
    for i, blk in enumerate(packed, start=1):
        blk["ref"] = i

    return {
        "blocks": packed,               # 已编号，可直接拼 prompt
        "dropped": dropped,             # 被预算挤掉的，可用于日志/告警
        "used_tokens": budget - remaining,
        "budget": budget,
        "utilization": (budget - remaining) / budget,
    }


def render_prompt(packed: Dict, query: str, system: Optional[str] = None,
                  strict: bool = False) -> str:
    """把装填结果渲染成最终 prompt。

    输出结构固定为三段：系统提示词 → 参考资料 → 问题。
    为什么资料段要用 `[n] 来源：路径/标题` 开头：这样每一块都自带出处，
    LLM 被要求"标注引用"时才知道该写哪个编号，人工核查时也能直接定位文件。

    system / strict 的默认值刻意保持「与加入生成层之前完全一致」——
    render_prompt 已被 stress_test 等外部调用依赖，默认行为不能漂移。
    strict=True 时才注入 config.ANSWER_SYSTEM_PROMPT（含强制 [n] 引用约束），
    供 pipeline.answer() 使用。
    """
    if system is None:
        if strict and USE_CITATION_CONSTRAINT:
            system = ANSWER_SYSTEM_PROMPT
        else:
            system = "你是严谨的文档助手，请仅依据给定资料回答，并标注引用编号。"
    if not packed["blocks"]:
        # 没有资料时也要有明确标记：LLM 看到"未检索到相关资料"才可能正确地拒答，
        # 而不是凭自己的知识编一个答案（这正是幻觉的高发场景）。
        context = "（未检索到相关资料）"
    else:
        # 用空行分隔各块：让模型能清楚区分"这是一块资料"的边界，
        # 避免把两块内容连读成一句话而误解。
        context = "\n\n".join(
            f"[{b['ref']}] 来源：{b.get('source')}"
            f"{(' / ' + b['heading']) if b.get('heading') else ''}\n{b['text']}"
            for b in packed["blocks"])
    return f"{system}\n\n===== 参考资料 =====\n{context}\n\n===== 问题 =====\n{query}"
