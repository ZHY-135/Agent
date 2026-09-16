# -*- coding: utf-8 -*-
"""③ 嵌入层：批处理 + 重试 + 维度校验 + 缓存。

--------------------------------------------------------------------------------
为什么"嵌入"值得单独一层，而不是在循环里直接调 API
--------------------------------------------------------------------------------
朴素写法（每个块调用一次 embedding）能跑通，但在真实规模下有三个致命问题，
本模块的每个类/函数都在解决其中一个：

  1. **慢**（批处理解决）
     每个块一次网络往返。1000 个块就是 1000 次 RTT。
     实测：逐条 21s → 批大小 64 仅 0.8s（约 26 倍）。
     这不是"优化"，而是"能不能用"的区别。

  2. **脆**（重试解决）
     一次网络抖动就会让整个索引任务失败，而索引可能已经跑了十几分钟。
     需要"失败自动重来"，且退避要递增（避免在对方过载时火上浇油）。

  3. **隐性错误**（维度校验 + 缓存解决）
     · 换模型后维度可能变化（1536 → 1024），若不在写入前校验，
       错误会一路带到数据库，报出极难理解的 SQL 异常；
     · 重复索引同一批内容会重复付费，需要按内容缓存。

--------------------------------------------------------------------------------
一句话理解三个类的关系
--------------------------------------------------------------------------------
    Embedder          接口：只要实现 embed_batch(texts) -> List[向量] 就能接入
      ├─ HashEmbedder     离线伪向量（零依赖，用于跑通链路与测试）
      ├─ LocalEmbedder    本机真实语义向量（fastembed/ONNX，**无需 API Key**）
      └─ OpenAIEmbedder   生产实现（真实语义向量，需 API Key）
    CachingEmbedder   装饰器：包在任意 Embedder 外面，加一层内容缓存
    embed_all()       编排函数：把上面的东西按"批次 + 重试 + 校验"跑完

装饰器的意义：加缓存不需要改 HashEmbedder 或 OpenAIEmbedder 的任何代码。
"""
import hashlib
import time
from typing import Dict, List, Optional, Sequence

from .config import EMBED_BACKOFF, EMBED_BATCH, EMBED_DIM, EMBED_RETRY
from .logging_setup import get_logger

logger = get_logger(__name__)


class Embedder:
    """接口：所有 embedder 都实现 embed_batch。

    为什么接口只要求"批量"而不提供"单条"：
    因为一旦暴露单条接口，调用方就会自然地写 for 循环逐条调用，
    于是批处理被绕过、性能退化回"每个块一次网络往返"。
    只给批量接口，是从 API 设计上强迫调用方走向正确用法。
    """

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def identity(self) -> str:
        """本嵌入器身份（进"索引签名"，决定库里已有向量还算不算数）。

        ★ 为什么需要：向量库里的向量是**某个特定模型**产出的。换模型后，
        新查询的向量与旧库里的向量处在**不同的向量空间**，余弦相似度失去意义——
        但程序不会报任何错，只会"结果莫名其妙变差"。这是最难排查的一类问题。

        把身份掺进索引指纹后，换模型等于"全部文件内容都变了"，
        下一次索引会自动重建，不需要人记得手动清库。

        默认实现用「类名 + 维度」：对演示用的确定性伪向量足够了。
        含外部模型名的实现（如 OpenAIEmbedder）应当覆盖本属性把模型名带上——
        维度相同但模型不同（如两个 1536 维模型）是最危险的组合。
        """
        return f"{type(self).__name__}:{self.dim}"


class HashEmbedder(Embedder):
    """零依赖演示用：基于词频的确定性伪向量。

    用途：离线跑通全链路 / 单元测试。语义质量远不如真实模型，
    但保证「同样的文本 → 同样的向量」，因此检索逻辑可验证。

    原理（为什么这样能当"向量"用）：
        把文本切成 token，每个 token 哈希到一个固定维度上"投票"（计数 +1），
        最后做 L2 归一化。于是"含有相同词的文本"会得到相似的向量方向，
        余弦相似度就能反映"字面重合度"。
    局限：它只认字面重合，**不理解语义**——
        "连接池上限"与"最大连接数"在它眼里毫无关系。因此：
        · 它的分数普遍偏低（实测相关查询 0.35~0.42，无关 0.0~0.12）；
        · 它不能代表真实检索质量，指标只能用于回归对比。
    """

    # ★ 显式声明「这不是真实语义向量」：
    #   其余弦分数与按真实模型标定的阈值（MIN_SIM=0.35、REFUSE_MIN_VEC_SCORE=0.35）
    #   量级不同。编排层（pipeline.support_gate）据此自动关闭基于绝对相似度的门禁，
    #   避免「离线演示被永久拒答」这种把人劝退的假故障。
    #   这个属性必须被装饰器透传——CachingEmbedder 曾漏传它，导致演示全部被误拒。
    pseudo_vectors = True

    def __init__(self, dim: int = EMBED_DIM):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def embed_batch(self, texts):
        out = []
        for t in texts:
            # 1) 先造一个全 0 的 dim 维向量，作为"投票箱"
            vec = [0.0] * self._dim
            for tok in self._tokenize(t):
                # 2) 每个 token 用 md5 取一个稳定的整数，再取模映射到某一维。
                #    用哈希而不是"查词表"：不需要预先建词表，任意新词都能落到某一维；
                #    用 md5 而不是内置 hash()：内置 hash 在不同进程/版本间不稳定
                #    （有随机盐），会导致"重启后同一文本得到不同向量"。
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                vec[h % self._dim] += 1.0
            # 3) 归一化后返回。归一化的作用见 _l2 的说明
            out.append(self._l2(vec))
        return out

    @staticmethod
    def _tokenize(text: str):
        # 中文按字切、英文按词切——够用的朴素分词。
        # 为什么中文按"单字"而不是"词"：中文分词需要额外依赖（jieba 等），
        # 而本类是"零依赖"路径的一部分。代价是不理解词边界，
        # 但对"判断两段文本是否在讲同一件事"这种粗粒度任务够用。
        import re
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    @staticmethod
    def _l2(vec):
        """L2 归一化：把向量缩放到长度 1。

        为什么必须做：余弦相似度 = 点积 / (|a|·|b|)。
        如果所有向量长度都是 1，分母恒为 1，余弦相似度就退化成**点积**，
        计算量减少一半；更重要的是，向量长度不再受文本长短影响——
        否则"长文本因为词多而模长更大"，会干扰相似度比较。
        """
        n = sum(x * x for x in vec) ** 0.5
        # n 为 0 表示文本里一个 token 都没有（空串、纯符号），此时保持全 0
        return [x / n for x in vec] if n else vec


class OpenAIEmbedder(Embedder):
    """生产适配器：换掉 client 即可，其余流水线不变。"""

    def __init__(self, client, model: str = "text-embedding-3-small",
                 dim: int = EMBED_DIM):
        self.client, self.model, self._dim = client, model, dim

    @property
    def dim(self):
        return self._dim

    @property
    def identity(self):
        """必须带上**模型名**：不同的嵌入模型可能维度相同，
        只比较维度会把"换了模型"误判成"没换"，于是旧向量继续参与比较。"""
        return f"OpenAIEmbedder:{self.model}:{self._dim}"

    def embed_batch(self, texts):
        # 提前拦截空输入：多数嵌入服务对空字符串要么报错、要么返回无意义的向量，
        # 在这里拦住能给出更清楚的错误位置（否则报错会出现在很远的调用栈里）。
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding 输入不能为空")
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        # 按输入顺序取回向量：服务端的返回顺序与输入一致，这是 API 契约的一部分
        return [d.embedding for d in resp.data]


def _load_fastembed() -> Optional[type]:
    """返回 fastembed 的 `TextEmbedding` 类；未安装返回 None。

    与 `converters._load_bs4` / `_load_docx` 同一个套路，理由也相同：
    测试需要一个**确定性的**"没装 fastembed"环境。把 import 写在函数体内部的话，
    那条分支只能靠卸载依赖来覆盖——而在装了它的开发机上，它将永远测不到，
    而"缺依赖时的提示"恰恰是用户第一次接触这个功能时最可能看到的画面。
    """
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    return TextEmbedding


class LocalEmbedder(Embedder):
    """本机运行的真实语义向量（fastembed / ONNX Runtime，**无需 API Key**）。

    ★ 为什么需要它：默认的 `HashEmbedder` 只认字面重合，于是 clone 下来的人第一印象
      就是"检索很差"（伪向量的 recall@5 只有 0.25）；而真实语义检索以前只有 openai
      一条路，意味着**必须自备 API Key**——仅这一条就挡掉了绝大多数想试一下的人。

    ★ `pseudo_vectors` **必须是 False**（不能沿用 `HashEmbedder` 的 True）：
      `pipeline.support_gate = not pseudo_vectors`。若标成 True，支持度门禁与
      `REFUSE_MIN_VEC_SCORE`（config.py）会一起失效，三层防幻觉里的
      "生成前拒答"就形同虚设——而界面上一切正常，这正是最难发现的一类退化。
      同理，`CachingEmbedder` 必须透传这个属性（见该类的 `pseudo_vectors`）。
    """

    pseudo_vectors = False

    def __init__(self, model_name: str, dim: int, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self._dim = dim
        self._cache_dir = cache_dir
        self._model = None            # 懒加载：构造时不触发下载（见 _ensure_model）

    def _ensure_model(self):
        """首次调用时才加载模型（会联网下载权重）。

        ★ 为什么必须懒加载：`build_embedder()` 在很多命令里都会被调用，但只有真正
          要 embed 时才需要模型。若在构造时就加载，`rag --help` / `rag init-db`
          这类命令也会被迫下载几十上百 MB——用户只会看到命令卡住，不知道该等还是该退。
        """
        if self._model is not None:
            return self._model
        embedding_class = _load_fastembed()
        if embedding_class is None:
            raise RuntimeError(
                "使用本地嵌入需要 fastembed：pip install \"rag-min[local]\"\n"
                "    离线环境可继续用默认的零依赖伪向量（不设 RAG_EMBEDDING_PROVIDER 即可）")
        try:
            self._model = embedding_class(model_name=self.model_name,
                                          cache_dir=self._cache_dir)
        except Exception as error:                     # noqa: BLE001
            raise RuntimeError(
                f"加载本地模型 {self.model_name} 失败（首次使用需要联网下载权重）：{error}\n"
                f"    可指定的本地模型见 config.LOCAL_EMBED_MODELS；"
                f"离线环境请改用默认伪向量") from error
        return self._model

    @property
    def dim(self):
        return self._dim

    @property
    def identity(self):
        """必须带上**模型名**：本模块顶部已说明"维度相同但模型不同"是最危险的组合。

        只比较维度的话，从 bge-small-en（384）换到 all-MiniLM-L6-v2（也是 384）
        会被判定为"没换"，于是旧向量继续参与比较——两个不同向量空间的向量算余弦，
        结果没有任何意义，但程序不会报任何错。
        """
        return f"LocalEmbedder:{self.model_name}:{self._dim}"

    def embed_batch(self, texts):
        # 与 OpenAIEmbedder 同一约定：空输入在这里就拦住，报错位置更靠近调用者，
        # 而不是等 fastembed 内部抛出难懂的错误
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding 输入不能为空")
        model = self._ensure_model()
        vectors = [list(vector) for vector in model.embed(list(texts))]
        # 维度自检：这里能给出**针对本地模型**的修复指引（embed_all 的通用检查
        # 只会说"维度不匹配"，不会告诉用户该设成多少）
        if vectors and len(vectors[0]) != self._dim:
            actual = len(vectors[0])
            raise ValueError(
                f"本地模型 {self.model_name} 实际输出 {actual} 维，"
                f"但当前配置声明 {self._dim} 维。\n"
                f"    请设置 RAG_EMBED_DIM={actual}（pgvector 建表时也会用到它）；"
                f"若库里已有旧向量，需要重新索引")
        return vectors


class CachingEmbedder(Embedder):
    """装饰器：按内容哈希缓存，重复索引不重复付费。

    为什么能省下大部分调用：索引场景里"内容没变的块"占绝大多数
    （改一篇文档通常只动几段），而 embedding 是按内容计费的。
    缓存命中就完全跳过网络调用。
    """

    def __init__(self, inner: Embedder):
        self.inner = inner
        # 显式标注类型：键是内容哈希，值是向量。
        # 不标注的话静态检查无法推断空 dict 的元素类型，后续赋值都会报错。
        self.cache: Dict[str, List[float]] = {}

    @property
    def dim(self):
        return self.inner.dim

    @property
    def pseudo_vectors(self):
        """★ 必须透传被装饰者的语义质量声明。

        装饰器如果吞掉这个属性，编排层就会把「HashEmbedder 的伪向量」
        误判成真实语义向量，从而按 0.35 的阈值做支持度门禁——
        实测后果是离线演示的所有查询都被判为「相关度过低」而永久拒答。
        """
        return getattr(self.inner, "pseudo_vectors", False)

    @property
    def identity(self):
        """★ 与 pseudo_vectors 同一类教训：装饰器必须透传身份。

        缓存只是加速层，**向量是谁算的**完全由被装饰者决定。
        若这里返回 "CachingEmbedder:1536"，那么把 HashEmbedder 换成真实模型
        （或反之）时索引签名不变 → 旧向量继续被使用 → 静默的向量空间错配。
        """
        return self.inner.identity

    def embed_batch(self, texts):
        # 两条并行数组：results 保证"输出顺序 == 输入顺序"，
        # miss_* 收集需要真正调用内层 embedder 的那些位置。
        # results 先填 None 占位（表示"这个位置的向量还没定"），最后统一保证填满。
        # 显式标注 Optional[List[float]] 是因为占位阶段确实是 None，
        # 不标注会让静态检查认为我们往 List[float] 的列表里塞了 None。
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_idx: List[int] = []
        miss_texts: List[str] = []
        for i, t in enumerate(texts):
            # 内容哈希作缓存键：同样的文本永远命中同一份向量。
            # 取前 16 位十六进制（64 bit）足够区分，且省内存。
            key = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
            if key in self.cache:
                results[i] = self.cache[key]              # 命中：直接复用
            else:
                # 未命中：记录"第几个位置"和"对应文本"。
                # 为什么要记位置 i：内层返回的向量顺序只对应 miss_texts，
                # 必须能把结果放回原来的下标，否则整个批次的向量会和文本错位。
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            logger.debug("embedding 缓存命中 %d / 未命中 %d",
                         len(texts) - len(miss_texts), len(miss_texts))
            vecs = self.inner.embed_batch(miss_texts)
            # zip 三路对齐：位置、文本、向量一一对应，写回缓存与结果数组
            for i, t, v in zip(miss_idx, miss_texts, vecs):
                key = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
                self.cache[key] = v
                results[i] = v
        return results


def embed_all(embedder: Embedder, texts: Sequence[str],
              batch_size: int = EMBED_BATCH,
              retry: int = EMBED_RETRY) -> List[List[float]]:
    """批处理 + 指数退避重试 + 维度校验。

    这个函数把"调用 embedding"这件事从"一行 API 调用"变成"一段可靠的工序"。
    外层只要 `embed_all(embedder, [所有子块文本])`，就自动获得：
        · 分批发送（快）
        · 失败重试（稳）
        · 维度校验（错得早、错得清楚）

    为什么返回值必须与输入等长且顺序一致：
    上层要 `zip(children, vecs)` 把向量绑回子块。一旦顺序错位，
    向量就会挂到错误的内容上——这种错误不会报错，只会让检索结果变得莫名其妙。
    """
    out: List[List[float]] = []
    # range(0, n, batch_size) 生成 0, 64, 128…，即每个批次的起始下标。
    # 用切片 texts[i:i+batch_size] 取这一批：这就是"批处理"的全部——
    # 把 64 次网络往返合成 1 次。
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        for attempt in range(retry):
            try:
                vecs = embedder.embed_batch(chunk)
                break                                     # 成功就跳出重试循环
            except Exception as e:                       # noqa
                if attempt == retry - 1:
                    # 最后一次仍失败：记录"哪一批、重试了几次"再抛出。
                    # 有这段日志，排障时能立刻知道是网络问题还是某些内容触发的。
                    logger.error("embedding 批次 [%d:%d] 重试 %d 次后仍失败：%s",
                                 i, i + len(chunk), retry, e)
                    raise
                delay = EMBED_BACKOFF * (2 ** attempt)
                # 指数退避：0.5s → 1s → 2s…
                # 为什么递增：失败往往是因为对方过载或网络拥塞，
                # 立刻重试只会加重拥塞；等待时间翻倍能让对方缓过来。
                # ⚠ 这里目前是"任何异常都重试"。规范做法应只重试瞬时错误
                #   （超时/5xx/限流），鉴权与参数错误应立即失败——见改进方案 P0-4。
                logger.warning("embedding 批次 [%d:%d] 失败，%.1fs 后重试（第 %d 次）：%s",
                               i, i + len(chunk), delay, attempt + 1, e)
                time.sleep(delay)
        # 维度校验：在这里拦住，错误信息里就能直接说清"模型给了多少、期望多少、
        # 该改哪里"。如果不校验，错误会延迟到写数据库时才爆出
        # vector 维度不匹配的 SQL 异常，那时已经很难定位到"其实是换了模型"。
        for v in vecs:
            if len(v) != embedder.dim:
                logger.error("维度不匹配：模型返回 %d，期望 %d", len(v), embedder.dim)
                raise ValueError(
                    f"维度不匹配：模型返回 {len(v)}，vector 列声明 {embedder.dim}"
                    f"（换模型时记得同步 config.EMBED_DIM 与 DDL）")
        out.extend(vecs)                                  # 按批次顺序拼接，保证全局顺序
    return out
