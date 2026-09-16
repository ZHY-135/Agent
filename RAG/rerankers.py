# -*- coding: utf-8 -*-
"""⑥ 重排层：MMR 多样性 + CrossEncoder 接口。

--------------------------------------------------------------------------------
这一层要解决什么问题
--------------------------------------------------------------------------------
检索排出的前几名，往往是"内容高度重复"的几段。原因有两条：
    · 子块之间有滑窗重叠（相邻块共享一部分文字）
    · 同一主题的多个子块都会命中同一个查询

于是 top-5 里可能有 3 条在说同一件事——**浪费了宝贵的上下文预算**：
LLM 的窗口就那么大，塞 3 条重复内容等于只得到 1 条信息量。

MMR（Maximal Marginal Relevance，最大边际相关性）的解法很直接：
选下一名时，同时考虑"它和问题有多相关"与"它和已选内容有多重复"。

--------------------------------------------------------------------------------
两类重排器的分工（不要混淆）
--------------------------------------------------------------------------------
    MMR（本模块的 mmr 函数）        —— 去冗余：让结果"彼此不同"
    CrossEncoderReranker           —— 精排：让结果"更准"

MMR 不判断"内容对不对"，它只看方向是否重复；
CrossEncoder 把 (查询, 文档) 成对送进模型，直接输出相关性分数，比向量余弦准得多，
但**代价高**（每对都要过一次模型，无法预计算），所以只用于对少量候选精排。

标准组合是"粗排召回 50 → CrossEncoder 精排 → 取 5"。
本项目当前只接了 MMR，CrossEncoder 仅提供接口（见改进方案 F4-4）。
"""
import math
from typing import Dict, List

from .config import MMR_LAMBDA


def mmr(hits: List[Dict], vectors: Dict[str, List[float]],
        top_k: int, lam: float = MMR_LAMBDA) -> List[Dict]:
    """最大边际相关性：在「相关」与「不重复」之间取平衡。

    公式：MMR = argmax [ λ·sim(q, d) − (1−λ)·max sim(d, 已选) ]
        第一项 λ·rel          —— 奖励"与查询相关"
        第二项 (1−λ)·redundancy —— 惩罚"与已选内容相似"
        λ=1  → 只看相关性（等价于按分数排序）
        λ=0  → 只看多样性（会选出最不相似的，通常没用）
        默认 0.7 偏重相关性，多样性作为调节项

    执行过程是一个"贪心迭代"（下面 while 循环）：
        每轮遍历所有剩余候选，为每个候选算出 MMR 值（需要与**所有已选**比较取最相似的那个），
        挑出 MMR 最大的加入已选，再从候选池移除，直到选满 top_k。
    为什么用贪心而不是全局最优：全局最优是组合优化问题（阶乘级），
    贪心是它的标准近似，效果足够好且复杂度可接受。

    ★ 量纲问题（本项目真实事故，务必理解）：
        rewards 来自 score，而 score 可能是 RRF 融合分（≈0.015 量级）。
        惩罚项来自余弦相似度（[0,1] 量级）。两者直接相减时，
        惩罚项会**完全压倒**奖励项，导致 MMR 退化成"只选最不相似的"——
        结果不会报错，只是悄悄变差。所以下面先把 rel 归一化到 [0,1]。
    """
    if not hits:
        return []

    # ★ 尺度对齐：把 score 线性缩放到 [0,1]，使其与余弦相似度可比。
    #   用 min-max 归一化而不是除以最大值：除以最大值不保证下界为 0，
    #   当所有分数都很接近时（RRF 分的典型情况）尺度仍会失衡。
    scores = [h["score"] for h in hits]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0          # 所有分数相同时 span=0 → 用 1.0 避免除零
    # 用 id(h) 作键而不是 child_id：同一个 Python 对象在一次调用内 id 唯一，
    # 且不要求 hit 里必须有 child_id（便于测试传入最小字典）。
    rel_of = {id(h): (h["score"] - lo) / span for h in hits}

    selected: List[Dict] = []
    pool = list(hits)                # 复制一份，避免就地修改调用方传入的列表
    while pool and len(selected) < top_k:
        best, best_val = None, -1e9  # 用一个足够小的初值，任何真实 MMR 值都会胜出
        for h in pool:               # 遍历所有剩余候选，找出本轮 MMR 最大的
            rel = rel_of[id(h)]
            # 冗余度 = 该候选与"已选中的每一个"的最大相似度。
            # 取 max 而不是平均：只要和其中一个高度重复，就该被惩罚
            #（平均值会被"与其他都不相似"稀释，从而放过重复内容）。
            # default=0.0：第一轮 selected 为空，此时没有冗余可言。
            red = max((_cos(vectors.get(h["child_id"], []),
                            vectors.get(s["child_id"], []))
                       for s in selected), default=0.0)
            val = lam * rel - (1 - lam) * red
            if val > best_val:
                best, best_val = h, val
        # pool 非空时循环至少执行一次，best_val 初值 -1e9 保证第一次必然赋值。
        # 显式断言而不是忽略告警：万一将来有人把初值改成正数、或让 pool 变空，
        # 这里会立刻炸掉，而不是把一个 None 塞进结果里。
        assert best is not None, "MMR 内部错误：候选池非空却没有选出任何候选"
        selected.append(best)
        pool.remove(best)            # 选过的从候选池移除，避免重复选中
    return selected


def _cos(a, b):
    """余弦相似度。用于比较两个子块向量的方向是否一致（即内容是否重复）。

    任一向量为空时返回 0.0：调用方（mmr）取不到向量时应该"认为不重复"，
    而不是抛异常——重排是可选增强，不应该因为缺向量就让整个检索失败。
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class CrossEncoderReranker:
    """生产重排器接口：用 cross-encoder 精排 top_k*4 候选。

    注意：重排必须在向量召回之后——它只救「召回但排错序」，
    救不回「根本没召回」（那是切分与召回率的问题）。

    与 MMR 的关键区别：MMR 只能基于"已有信息"（向量、分数）做筛选；
    CrossEncoder 能**同时看查询和文档**，因此能判断"这段话是否真的回答了这个问题"。
    这也解释了 why 它更准也更慢。
    """

    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, hits: List[Dict], top_k: int) -> List[Dict]:
        """把 (查询, 文档) 成对送进模型打分，按新分数重排。

        为什么用 child_text 而不是父块正文：child 才是参与检索、与查询
        在同一粒度上可比的那段文本；用父块会引入大量与问题无关的内容，
        反而干扰模型判断。
        """
        pairs = [(query, h["child_text"]) for h in hits]
        scores = self.model.predict(pairs)
        # 把模型分数**写回** hit 对象：否则下游（预算装填、评测报告）看到的
        # 还是旧的融合分，与当前排序不自洽。
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        # 重排后必须自己截断：模型给的是全量候选的分数，
        # 不截断会把 top_k*4 个候选都返回给下游
        return sorted(hits, key=lambda x: -x["score"])[:top_k]
