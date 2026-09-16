# -*- coding: utf-8 -*-
"""⑤ 检索层：向量 + BM25 混合 → RRF 融合 → 父子聚合。

--------------------------------------------------------------------------------
为什么单一向量检索不够（这一层存在的理由）
--------------------------------------------------------------------------------
向量检索擅长"意思相近"，但**区分不开字面精确的标识符**。
原因是嵌入模型会把语义平滑：一段文档里 "0x80070005" 这几个字符
对整体向量方向的影响微乎其微。于是当语料里有多段内容语义相似时，
向量给出的分数极差极小——**不是"找不到"，而是"分不开"**：

    语料里 5 段都在讲连接池，只有 1 段含错误码 0x80070005。
    向量通道：正确文档排第 1，但前 3 名分差极小（无法据此判断哪段才对）
    BM25 通道：只有含该 token 的文档得分 > 0，区分度是数量级的

所以两路通道是**互补**的，而不是"谁更好"：
    向量负责「语义召回」——用户换个说法也能找到
    BM25 负责「精确区分」——错误码、API 名、配置项名等专有 token

--------------------------------------------------------------------------------
两路分数不能直接相加，所以用 RRF（本层最关键的决策）
--------------------------------------------------------------------------------
余弦相似度落在 [0,1]，而 BM25 分数无上界（可以是 3.7，也可以是 42）。
把两者归一化后加权，需要为每个语料调参，且极不稳定。
RRF（Reciprocal Rank Fusion）换了个思路：**只看排名，不看分数**。

    最终分数 = Σ 权重 × 1/(k + 排名)

排名第 1 得 1/(60+1)、第 2 得 1/(60+2)……于是：
    · 不同量纲的问题自动消失（只用排名，天然可比）
    · 无需任何调参（k 取 60 是经验值，越靠前越占优）
    · 对单个通道的"异常高分"不敏感（鲁棒）
代价：丢掉了"分差信息"（第 1 名比第 2 名领先多少不再体现）。
对本场景而言，这个代价远小于量纲对齐带来的麻烦。

--------------------------------------------------------------------------------
Reader's guide
--------------------------------------------------------------------------------
    BM25.search()            —— 关键词通道：词频 + 逆文档频率
    rrf_fuse()               —— 把两路的"排名列表"融合成一个排名
    aggregate_by_parent()    —— 同一父块的多个子块命中只算一条（★ 顺序不能反）
"""
import math
import re
from typing import Dict, List, Optional, Sequence

from .config import RRF_K, SCORE_BM25, SCORE_KIND_KEY, SCORE_RRF


class BM25:
    """极简 BM25：关键词通道，补向量检索的短板。

    BM25 的核心直觉（为什么它能反映"相关性"）：
        一个词对某文档的重要性 ≈ 它在该文档出现的频率（TF） × 它在全库有多稀有（IDF）
        再对"文档太长导致词频虚高"做惩罚（长度归一化）

    举例：
        "0x80070005" 在某文档出现 1 次，但在全库 1000 个文档里只出现 1 次
        → IDF 极高 → 这个词几乎是这段文档的"指纹"，得分远超其他词
    这正是向量通道做不到的"精确区分"。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # k1 控制"词频饱和"速度：出现 10 次不该比出现 1 次重要 10 倍
        # b  控制长度归一化的强度：0=不归一化，1=完全归一化
        # 这两个是 BM25 的标准经验值，一般无需调整
        self.k1, self.b = k1, b
        self.docs: List[List[str]] = []
        self.ids: List[str] = []
        self.meta: Dict[str, Dict] = {}     # child_id → 原始记录（用于回填字段）
        self.avg_len = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # 与 HashEmbedder 用同一套切法（中文按字、英文按词），
        # 保证"关键词通道"和"向量通道"看到的是同一批字面单位，便于对照分析
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    def fit(self, rows: Sequence[Dict]):
        """建立词频统计。rows: 含 child_id / child_text / parent_id 的记录列表。

        这是"离线统计"阶段：把所有文档扫一遍，算出 IDF 与平均长度。
        必须先把全库看一遍才能算 IDF（IDF 的定义就是"这个词在全库有多稀有"），
        所以 BM25 无法只靠单条记录在线计算——这也是它需要重建索引的原因。
        """
        self.ids = [r["child_id"] for r in rows]
        # ★ 保存原始记录：search() 之后要回填 parent_id 等字段。
        #   BM25 只算分数，本身不知道 parent_id；如果 search() 只返回 (分数, 下标)，
        #   下游的 aggregate_by_parent() 会因取不到 parent_id 而 KeyError。
        #   （这是本项目真实踩过的坑：notebook §5.2）
        self.meta = {r["child_id"]: r for r in rows}
        self.docs = [self.tokenize(r["child_text"]) for r in rows]
        self.avg_len = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        self.df.clear()
        for d in self.docs:
            # set(d) 去重：DF（文档频率）统计的是"有多少篇文档含此词"，
            # 而不是"此词总共出现几次"（后者是 TF）。同一篇里出现 10 次也只算 1 篇。
            for w in set(d):
                self.df[w] = self.df.get(w, 0) + 1
        n = len(self.docs)
        # IDF 公式：log(1 + (N - df + 0.5)/(df + 0.5))
        # 加 0.5 是平滑，避免除零；外层 log(1+…) 保证结果非负
        #（罕见词 IDF 大 → 区分度高；"的""是"这类满库都有的词 IDF ≈ 0 → 几乎不影响得分）
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5))
                    for w, c in self.df.items()}
        return self

    def search(self, query: str, top_k: int) -> List[Dict]:
        """给每个文档算 BM25 分，返回最高的 top_k 条。

        注意这里是"全库扫描"：为了能跨文档比较分数，
        BM25 必须对每个文档算一次分，没有倒排索引加速。
        当前规模（几百到几千块）完全够用；上万块时应改为数据库全文索引（见改进方案 P1-6）。
        """
        q = self.tokenize(query)
        out = []
        for i, d in enumerate(self.docs):
            if not d:
                continue                              # 空文档无法参与打分
            # 先统计这篇文档的词频（TF）：某词在此文档中出现几次
            tf: Dict[str, int] = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            score = 0.0
            for w in q:
                if w not in tf:
                    # 查询词没在这篇文档出现 → 此词对这篇文档没有贡献。
                    # 这里**不加分也不扣分**，所以"没出现任何查询词"的文档得 0 分，
                    # 下面的 score > 0 判断会把它过滤掉。
                    continue
                f = tf[w]
                # BM25 的单项得分公式：
                #   idf × (f × (k1+1)) / (f + k1 × (1 - b + b × 文档长度/平均长度))
                # 分母里的长度项：文档越长，同一个词频的"含金量"越低
                #（长文档本来就容易碰巧包含某个词）
                denom = f + self.k1 * (1 - self.b + self.b * len(d) / max(self.avg_len, 1))
                score += self.idf.get(w, 0.0) * (f * (self.k1 + 1)) / denom
            if score > 0:
                # 只收正分的文档：避免把"毫无关系"的文档塞进候选，
                # 否则它们会污染后续的 RRF 融合（排名靠后但仍在列表里就占掉一个名次）
                out.append((score, i))
        # 降序排序：BM25 分数是"越大越相关"
        out.sort(key=lambda x: -x[0])
        # ★ 回填完整字段（原始记录 + 新分数），使关键词通道的返回结构与
        #   向量通道一致。两路结构一致，才能一起进 rrf_fuse。
        #   量纲标记写 bm25：BM25 分没有跨库可比性（随语料长度/词频变化），
        #   绝不能拿去和余弦阈值比较——标记让这种误用在 support_score 里当场失败。
        return [dict(self.meta[self.ids[i]], score=s, **{SCORE_KIND_KEY: SCORE_BM25})
                for s, i in out[:top_k]]


def rrf_fuse(rank_lists: List[List[Dict]], k: int = RRF_K,
             weights: Optional[Sequence[float]] = None) -> List[Dict]:
    """倒数排名融合：不用归一化分数，直接按排名加权，鲁棒且无需调参。

    输入：若干"已排好序"的命中列表（如 [向量召回结果, 关键词召回结果]）
    输出：一个融合后的排名列表

    执行过程（对应下面三段）：
        ① 逐路遍历，按排名给每个 child 累加 1/(k+排名)
        ② 同一个 child 被多路命中时，分数会叠加 → 天然奖励"两路都认可"的结果
        ③ 按累加分数重排

    为什么"两路都命中"就该排更前：它同时满足"语义相近"和"字面精确"，
    可信度确实高于只被一路命中。
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    fused: Dict[str, float] = {}
    payload: Dict[str, Dict] = {}
    # zip(weights, rank_lists) 把"权重"和"那一路的结果"配对，
    # 于是加权融合不需要额外分支：权重为 0 就等于忽略那一路
    for w, lst in zip(weights, rank_lists):
        for rank, item in enumerate(lst, start=1):     # start=1：排名从 1 开始（1/(k+1)）
            cid = item["child_id"]
            fused[cid] = fused.get(cid, 0.0) + w * (1.0 / (k + rank))
            # setdefault：只记录**第一次**遇到的该 child 的完整记录。
            # 两路返回的记录字段略有差异（向量通道 score=余弦，关键词通道 score=BM25），
            # 保留先到的那个作为"载荷"即可；融合后的分数会覆盖它。
            payload.setdefault(cid, item)
    order = sorted(fused.items(), key=lambda x: -x[1])
    # 用 payload 里的原始字段 + 融合后的新分数组装结果，
    # 这样下游既能拿 parent_id 做聚合，也能看到融合分。
    # ★ 量纲标记改为 rrf：融合分 ≈1/(k+rank) ≈0.016，**与余弦差两个数量级**。
    #   它是"排名分"而不是"相似度"，只能用于排序与多样性，
    #   一旦被拿去和 REFUSE_MIN_VEC_SCORE(0.35) 比较，所有查询都会被判"支持度不足"。
    #   本项目已经真实发生过同类事故（MMR 量纲不对齐），故在此显式标记。
    return [dict(payload[cid], score=s, **{SCORE_KIND_KEY: SCORE_RRF})
            for cid, s in order]


def aggregate_by_parent(hits: List[Dict]) -> List[Dict]:
    """★ 先按父块聚合（每个父块只留最高分子块），再去重——顺序不能反。

    为什么需要聚合：一个父块通常被切成多个子块，检索时可能命中其中好几个。
    但最终 prompt 里父块只会出现**一次**，所以候选里同一父块出现多次没有意义，
    留着会挤占 top_k 名额。

    为什么必须"先聚合后截断"（这是本项目真实踩过的坑）：
        反例：top_k=3 时若先截断，取到的前 3 个子块可能全属于同一个父块，
              聚合去重后只剩 1 条结果 —— 用户问一个问题，只拿到一段资料。
        正确顺序：先把所有候选按父块聚合，再截断，才能保证 3 个名额对应 3 个不同父块。

    实现细节：用 dict 按 parent_id 归组，只在遇到更高分时替换，
    最后按分数排序。这样既完成聚合，也完成"同一父块保留最高分"。
    """
    best: Dict[str, Dict] = {}
    for h in hits:
        pid = h["parent_id"]
        # 首次遇到该父块 → 直接放入；已存在且当前分更高 → 替换
        if pid not in best or h["score"] > best[pid]["score"]:
            best[pid] = h
    return sorted(best.values(), key=lambda x: -x["score"])
