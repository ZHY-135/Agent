# -*- coding: utf-8 -*-
"""全局配置：所有可调参数集中在此，避免散落各模块。"""

# ---------- ① 切分 ----------
MAX_PARENT_CHARS = 1200     # 父块上限：交付给 LLM 的上下文粒度
MAX_CHILD_CHARS  = 300      # 子块上限：embedding 的匹配粒度
CHILD_OVERLAP    = 50       # 滑窗重叠：child 的 10%~20%
MIN_CHILD_CHARS  = 60       # 尾块合并阈值
MERGE_RATIO      = 1.25     # 尾块合并后允许的超限倍数
FENCE            = "```"
# 中英文标点合并；⚠ lookbehind 必须固定宽度，不能写 ["\')\]]?
SENT_END = r'(?<=[。！？；!?;．])|(?<=[.!?;])(?=\s)'

# ---------- ② 嵌入 ----------
EMBED_DIM     = 1536        # text-embedding-3-small 的常用维度；换模型必须同步
EMBED_BATCH   = 64          # 批处理大小（实测 1000 块：逐条 21s → 批 64 仅 0.8s）
EMBED_RETRY   = 3           # 失败重试次数
EMBED_BACKOFF = 0.5         # 指数退避基数（秒）

# ---------- ③ 存储 ----------
HNSW_M              = 16    # 图连接数
HNSW_EF_CONSTRUCTION= 64    # 建索引质量
HNSW_EF_SEARCH      = 40    # 查询召回率/速度权衡

# ---------- ④ 检索 ----------
TOP_K         = 5           # 最终返回条数
CANDIDATE_MUL = 4           # 候选倍数：先取 top_k*4 再精排
MIN_SIM       = 0.35        # 相似度阈值，低于此不返回
BM25_WEIGHT   = 0.5         # 混合检索中关键词通道权重
RRF_K         = 60          # RRF 融合常数（k 越小，排名靠前越占优）

# ---------- ④b 分数量纲标记（内部契约，不是可调参数）----------
# 为什么需要它：本项目的分数有**三种互不可比的量纲**，而它们都是 "score"：
#     · 余弦相似度    [0,1]（向量通道、pgvector 的 1-distance）
#     · BM25 分       0~10 量级（关键词通道，数量级随语料变化）
#     · RRF 融合分   ≈1/(60+rank) ≈ 0.016 量级
# 已经真实踩过两次同类事故（MMR 量纲不对齐、拒答阈值拿融合分比较，见
# docs/ENGINEERING_REVIEW.md §8.3）：两者都表现为"不报错但结果全错"。
# 因此每条命中都带上自己是哪种量纲，越界使用的地方**直接抛错**，
# 把"静默算错"变成"当场失败"。
SCORE_KIND_KEY = "_score_kind"
SCORE_COSINE = "cosine"          # 只可与 MIN_SIM / REFUSE_MIN_VEC_SCORE 比较
SCORE_BM25 = "bm25"              # 只用于排名融合，绝对值无跨库可比性
SCORE_RRF = "rrf"                # 只用于排名/多样性，绝不可当相似度用

# ---------- ⑤ 重排 ----------
MMR_LAMBDA    = 0.7         # 相关性 vs 多样性权衡（1=纯相关，0=纯多样）

# ---------- ⑥ 上下文 ----------
CONTEXT_BUDGET = 6000       # prompt 总 token 预算
CHARS_PER_TOKEN = 1.5       # 中文粗估：1.5 字符/token

# ---------- ⑦ 生成 ----------
USE_CITATION_CONSTRAINT = True    # 是否启用 [n] 引用强制约束与回填校验
MIN_ANSWER_CITATIONS   = 1        # 答案至少需要几个合法引用才算「有据」
REFUSE_ON_INVALID_CITATION = True # 引用了不存在的编号时拒答（凭空编号 = 最强幻觉信号）
REFUSE_ON_NO_CONTEXT   = True     # 检索为空时直接拒答，不调用 LLM
REFUSE_ON_WEAK_SUPPORT = True     # 向量支持度不足时拒答

# ⚠ 按「原始余弦相似度」标定，不是 RRF 融合分：
#   融合分 ≈ 1/(60+rank) ≈ 0.016 量级，与余弦差两个数量级，直接比较会让所有查询都触发拒答。
#   这与 MMR 那次量纲事故（docs/ENGINEERING_REVIEW.md §8.3）是同一类错误。
REFUSE_MIN_VEC_SCORE   = 0.35

GENERATE_RETRY         = 2        # LLM 调用失败重试次数（仅瞬时错误）
GENERATE_BACKOFF       = 0.8      # 重试指数退避基数（秒）

# 答案侧系统提示词：把「只依据资料 / 强制标注 [n] / 资料不足时明说」三条约束固化下来。
# 说明：检索侧无约束；仅当 pipeline.answer() 走严格引用模式时才使用。
ANSWER_SYSTEM_PROMPT = """你是严谨的文档助手。请遵守以下规则：
1. 只依据「参考资料」作答，禁止使用资料之外的知识，禁止推测或补全。
2. 每个事实性陈述后面必须标注来源编号，格式为 [1] 或 [1][2]。
3. 若参考资料不足以回答问题，直接回复「根据现有资料无法回答」，并说明还缺什么信息；不要编造。
4. 回答简洁，先给结论再给依据。"""

# 拒答话术：让 LLM 自由发挥是幻觉的来源之一，拒答文案由代码固化。
REFUSAL_MESSAGES = {
    "no_context":       "根据现有资料无法回答：未检索到与该问题相关的文档片段。",
    "weak_support":     "根据现有资料无法回答：检索到的内容与问题相关度过低，不足以支撑答案。",
    "not_supported":    "根据现有资料无法回答：未能找到可支撑该结论的资料出处。",
    "invalid_citation": "根据现有资料无法回答：模型给出的引用编号不存在，答案不可信。",
    "empty_answer":     "根据现有资料无法回答：模型未给出有效回答。",
    "generator_error":  "生成失败：调用生成模型时发生错误，请稍后重试或检查 API 配置。",
}

# ---------- ⑧ 记忆治理 ----------
IMPORTANCE_DEFAULT = 0.5    # 记忆重要度默认值
FORGET_DAYS        = 30     # 不活跃天数
FORGET_IMPORTANCE  = 0.3    # 低于此重要度可被遗忘
