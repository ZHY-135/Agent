# -*- coding: utf-8 -*-
"""④ 存储层：向量存储抽象 + 内存实现 + pgvector 实现。

--------------------------------------------------------------------------------
为什么要有"抽象"这一层（而不是直接用数据库）
--------------------------------------------------------------------------------
因为**开发与运行需要两种完全不同的存储**：

    MemoryStore   —— 进程内 dict。零依赖、毫秒级、可随意丢弃。
                     用于单元测试与离线演示：跑 200 个用例不该需要起一个数据库。
    PgVectorStore —— PostgreSQL + pgvector。持久化、有向量索引、支持并发。
                     用于真实运行：进程退出后索引不能消失。

上层（pipeline.py）只认 `VectorStore` 这组接口，因此**同一份编排代码**
既能跑内存模式也能跑数据库模式。这也是本项目"换向量库只改装配层一个函数"的原因。

--------------------------------------------------------------------------------
这一层承担的两件容易被低估的事
--------------------------------------------------------------------------------
1. **幂等写入**：`replace_document()` 用"先删后插"实现更新。
   配合切分层的内容派生 ID，"重复索引同一文档"不会产生重复行。
2. **级联清理**：pgvector 版本用外键 ON DELETE CASCADE，
   于是"删文档"只需删一行 source_document，父块子块由数据库自动收拾——
   这让删除逻辑极短，也不会漏删产生孤儿数据。

--------------------------------------------------------------------------------
Reader's guide
--------------------------------------------------------------------------------
    VectorStore       接口清单（想换 Milvus/Qdrant 就实现这些方法）
    match_root()      目录归属判断（曾因用错写法导致 sync 静默失效）
    MemoryStore       内存实现，也是理解接口语义的最佳样本
    pg_ddl()          建表语句生成（含 HNSW 向量索引）
    PgVectorStore     数据库实现（注意每个方法都遵循 取连接 → try → 归还 的模式）
"""
import math
import os
from typing import Dict, List, Optional, Sequence

from .config import (
    EMBED_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    MIN_SIM,
    SCORE_COSINE,
    SCORE_KIND_KEY,
)

__all__ = ["VectorStore", "MemoryStore", "PgVectorStore", "pg_ddl", "match_root"]


class VectorStore:
    """上层只依赖这些接口，不读取具体 Store 的内部属性。

    这组方法可以按用途分成三类：
        增量索引   get_document_fingerprint / list_sources / replace_document / delete_document
        （这些支撑"文件没变就跳过"和"磁盘删了就清索引"）
        检索       search_children / get_child_vectors
        回表       get_parents / list_children
        （检索命中子块后要取回父块全文，这就是"回表"）
    """

    def get_document_fingerprint(self, source: str) -> Optional[str]:
        """返回该文件上次索引时的内容指纹；None 表示还没索引过。"""
        raise NotImplementedError

    def list_sources(self, root: str) -> List[str]:
        """列出 root 目录下**已索引**的所有文件（用于清理已删除文件的残留索引）。"""
        raise NotImplementedError

    def replace_document(self, source: str, fingerprint: str,
                         parents: List[Dict], children: List[Dict]) -> None:
        """整体替换一个文档的索引内容（幂等：重复调用结果相同）。"""
        raise NotImplementedError

    def delete_document(self, source: str) -> None:
        raise NotImplementedError

    def search_children(self, query_vec, top_k: int,
                        min_sim: float = MIN_SIM) -> List[Dict]:
        """向量召回的入口：返回最相似的 top_k 个子块。"""
        raise NotImplementedError

    def list_children(self) -> List[Dict]:
        """取出全部子块（BM25 需要全量语料才能算 IDF，因此这个方法无法加分页）。"""
        raise NotImplementedError

    def get_child_vectors(self, child_ids: Sequence[str]) -> Dict[str, List[float]]:
        """按 ID 批量取子块向量（MMR 算多样性、生成层算支持度都要用）。"""
        raise NotImplementedError

    def get_parents(self, parent_ids: Sequence[str]) -> Dict[str, Dict]:
        """按 ID 批量取父块全文（"回表"：检索命中子块，但喂给 LLM 的是父块）。"""
        raise NotImplementedError


def match_root(source: str, root: str) -> bool:
    """判断 `source` 是否位于目录 `root` 之下（含 root 本身是文件的情况）。

    ★ 两个必须避免的写法（都曾真实出错）：

    1. **不要用 `os.path.commonpath([root, source]) == abspath(root)`**
       本类的 source 由 `loaders.iter_documents()` 用 `abspath` 生成，
       而 root 是调用方的**原始入参**。调用方传相对路径（CLI/README 的常规用法）时，
       Windows 上 commonpath 混合相对与绝对路径会直接抛
       `ValueError: Paths don't have the same drive`，异常一路上抛，
       导致 `index(sync=True)` 的清理逻辑被跳过、脏索引残留（deleted 恒为 0）。
       另外 commonpath 做前缀判断本身也是错的：
       `commonpath(['/a/docs2', '/a/docs/x'])` 返回 `/a`，会把兄弟目录算进来。

    2. **不要用裸 `str.startswith`**
       `"D:\\docs2\\f.md".startswith("D:\\docs")` 为 True —— 前缀相同但不是子目录。
       必须先补分隔符再比较。

    解决思路：两边都先 `abspath` 归一到同一形态（消除"相对 vs 绝对"的差异），
    再比较"是否以 root + 分隔符 开头"。
    Windows 路径大小写不敏感，故比较前统一 casefold；其他平台保持原样。
    """
    source_abs = os.path.abspath(source)
    root_abs = os.path.abspath(root)
    if os.name == "nt":
        # Windows 下 "D:\Docs" 与 "D:\docs" 是同一目录，必须统一大小写再比较
        source_abs, root_abs = source_abs.casefold(), root_abs.casefold()
    if source_abs == root_abs:
        return True                              # root 本身是文件（单文件索引）
    # 补分隔符再比较：这样 "docs2" 就不会被误判为 "docs" 的子目录
    prefix = root_abs.rstrip("\\/") + os.sep
    return source_abs.startswith(prefix)


class MemoryStore(VectorStore):
    """零依赖实现，供离线演示和单元测试使用。

    内部就是三个 dict，理解它们的分工就理解了数据模型：
        documents  source(文件路径) → 指纹           —— 增量索引用
        parents    parent_id        → 父块记录        —— 回表用
        children   child_id         → 子块记录(含向量) —— 检索用
    """

    def __init__(self):
        self.children: Dict[str, Dict] = {}
        self.parents: Dict[str, Dict] = {}
        self.documents: Dict[str, str] = {}

    def get_document_fingerprint(self, source):
        # .get 而不是 []：文件没索引过时返回 None（表示"需要索引"），
        # 用 [] 会抛 KeyError，调用方就得额外 try 包围
        return self.documents.get(source)

    def list_sources(self, root):
        """列出 root 之下的所有已索引文件（绝对路径）。见 `match_root` 的说明。"""
        return [source for source in self.documents if match_root(source, root)]

    def replace_document(self, source, fingerprint, parents, children):
        """先删后插 = 幂等更新。

        为什么不用"逐条比较再增删改"：
        那样要处理"哪些块新增、哪些消失、哪些只是顺序变了"三种情况，逻辑复杂且易错。
        "先删光旧数据、再插入新数据"只有一条路径，行为可预测；
        代价是重复插入，但对本地内存操作和事务内的数据库操作都不算问题。
        """
        self.delete_document(source)
        self.documents[source] = fingerprint
        for parent in parents:
            # dict(parent) 拷贝一份再存：避免调用方之后修改原对象时
            # 悄悄改掉存储里的内容（内存模式下两者本来会指向同一个 dict）
            self.parents[parent["parent_id"]] = dict(parent)
        for child in children:
            self.children[child["child_id"]] = dict(child)

    def delete_document(self, source):
        """删除文档及其所有父块、子块。

        步骤刻意分成两段：先找出该文档的所有父块 ID，
        再用这批 ID 反查并删除子块。顺序不能反——
        子块只记录 parent_id，不记录 source，所以必须先经由父块才能定位子块。
        （这正是 pgvector 版用外键级联来做同一件事的原因）
        """
        parent_ids = [pid for pid, parent in self.parents.items()
                      if parent.get("source") == source]
        for pid in parent_ids:
            self.parents.pop(pid, None)
        for cid in [cid for cid, child in self.children.items()
                    if child["parent_id"] in parent_ids]:
            self.children.pop(cid, None)
        self.documents.pop(source, None)

    def search_children(self, query_vec, top_k, min_sim=MIN_SIM):
        """暴力扫描：对每个子块算余弦相似度，排序取前 top_k。

        复杂度 O(N)——只适合演示与测试。真实规模应交给 pgvector 的 HNSW 索引
        （近似最近邻，复杂度接近 O(log N)）。
        这里的实现刻意保持"最直白"：没有索引、没有近似，便于验证检索逻辑本身是否正确。
        """
        scored = []
        for row in self.children.values():
            sim = self._cosine(query_vec, row["embedding"])
            if sim >= min_sim:
                # 阈值过滤在这里做（而不是事后再筛）：语义完全不相关的块
                # 不该进入候选，否则会挤占 top_k 名额并把噪声传给重排层
                scored.append((sim, row))
        scored.sort(key=lambda item: -item[0])
        # dict(row, score=score) 复制一份并附加分数，不修改存储中的原记录。
        # ★ 同时打上量纲标记：这里给出的是**原始余弦相似度**，
        #   只有余弦分才允许与 MIN_SIM / REFUSE_MIN_VEC_SCORE 这类绝对阈值比较。
        #   下游（support_score）会检查这个标记，防止有人误把 RRF 融合分当相似度。
        return [dict(row, score=score, **{SCORE_KIND_KEY: SCORE_COSINE})
                for score, row in scored[:top_k]]

    def list_children(self):
        # 同样复制：(dict(row) 是浅拷贝) 防止调用方改到存储内部状态
        return [dict(row) for row in self.children.values()]

    def get_child_vectors(self, child_ids):
        """只返回"存在且带向量"的那些 ID。

        用 dict 推导 + 条件过滤而不是直接取值：调用方传进来的 ID 里
        可能包含已被删除的块，此时应当跳过而不是抛 KeyError。
        """
        return {cid: self.children[cid]["embedding"] for cid in child_ids
                if cid in self.children and "embedding" in self.children[cid]}

    def get_parents(self, parent_ids):
        return {pid: self.parents[pid] for pid in parent_ids if pid in self.parents}

    @staticmethod
    def _cosine(a, b):
        """余弦相似度 = 点积 / 两个模长之积。

        为什么用余弦而不是欧氏距离：余弦只关心**方向**，不关心向量长度。
        文本越长向量模长越大，用欧氏距离会让"长文本天然更远"，
        余弦则消除了这个偏差。
        """
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        # na 或 nb 为 0（零向量）时返回 0.0 而不是除零崩溃
        return dot / (na * nb) if na and nb else 0.0


def pg_ddl(dim: int = EMBED_DIM) -> str:
    """生成与 embedding 维度一致的数据库 DDL。

    为什么维度要作为参数生成而不是写死：
    向量列的维度必须与嵌入模型的输出维度严格一致（1536 维的列塞不进 1024 维的向量）。
    把维度做成模板参数，就能保证"建表语句"和"嵌入配置"来自同一个值。

    表结构设计的三层父子关系：
        source_document（一个文件一行，存指纹）
            └─ parent_chunk（父块，外键指向 source_document）
                    └─ child_chunk（子块，外键指向 parent_chunk）
    外键都带 ON DELETE CASCADE：删除源文件那一行，父块子块会被数据库自动删除。
    """
    if dim <= 0:
        raise ValueError("embedding 维度必须为正数")
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS source_document (
        source TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS parent_chunk (
        parent_id UUID PRIMARY KEY,
        parent_text TEXT NOT NULL,
        source TEXT NOT NULL REFERENCES source_document(source) ON DELETE CASCADE,
        heading TEXT,
        parent_index INT,
        char_len INT
    );
    CREATE TABLE IF NOT EXISTS child_chunk (
        child_id UUID PRIMARY KEY,
        parent_id UUID NOT NULL REFERENCES parent_chunk(parent_id) ON DELETE CASCADE,
        child_text TEXT NOT NULL,
        embedding vector({dim}) NOT NULL,
        child_index INT,
        char_start INT,
        char_end INT
    );
    CREATE INDEX IF NOT EXISTS child_embedding_idx
        ON child_chunk USING hnsw (embedding vector_cosine_ops)
        WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
    CREATE INDEX IF NOT EXISTS child_parent_idx ON child_chunk (parent_id);
    CREATE INDEX IF NOT EXISTS parent_source_idx ON parent_chunk (source);
    """


class PgVectorStore(VectorStore):
    """psycopg2 + pgvector 实现。replace_document 在一个事务内完成。

    连接管理采用"连接池 + 取用/归还"模式，每个方法的结构都相同：
        conn = self._conn()      # 从池里取一个连接
        try:   ...               # 干活
        finally: self.pool.putconn(conn)   # ★ 无论成功失败都归还
    为什么用 try/finally 而不是 try/except：
    这里要保证的不是"异常被处理"，而是"连接一定回到池里"。
    忘记归还会让连接池逐渐枯竭，最终所有请求都卡在"等连接"上——
    这类故障现场离原因很远，所以必须在写法上就保证不会漏。
    """

    def __init__(self, conn_pool, dim: int = EMBED_DIM):
        self.pool, self.dim = conn_pool, dim

    def _conn(self):
        """取连接并注册 pgvector 类型适配。

        register_vector 的作用：告诉 psycopg2 如何在 Python list 与
        PostgreSQL 的 vector 类型之间转换。不调用它，传 list 会报类型错误。
        """
        from pgvector.psycopg2 import register_vector
        conn = self.pool.getconn()
        register_vector(conn)
        return conn

    def initialize(self):
        """建表 + 建索引。幂等：DDL 里全用了 IF NOT EXISTS，可重复执行。"""
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(pg_ddl(self.dim))
            conn.commit()                            # DDL 也需要提交
        except Exception:
            conn.rollback()                          # 失败要回滚，否则连接会留着半途状态
            raise
        finally:
            self.pool.putconn(conn)

    def get_document_fingerprint(self, source):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT fingerprint FROM source_document WHERE source = %s", (source,))
                row = cur.fetchone()
                # 只读查询无需 commit；没有记录时 fetchone 返回 None，
                # 正好表示"还没索引过"
                return row[0] if row else None
        finally:
            self.pool.putconn(conn)

    def list_sources(self, root):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                # 用 LIKE 前缀匹配而不是把全部记录取回 Python 过滤：
                # 让数据库做过滤，避免把全库路径读进内存。
                # ⚠ 已知局限：路径里的 % 与 _ 是 LIKE 的通配符且未转义，
                #   含这些字符的目录名会匹配出错（见改进方案 P0-1）。
                prefix = os.path.abspath(root).rstrip("\\/") + os.sep + "%"
                cur.execute("SELECT source FROM source_document WHERE source LIKE %s", (prefix,))
                return [row[0] for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def replace_document(self, source, fingerprint, parents, children):
        """在一个事务里完成"替换该文档的全部索引内容"。

        四步的顺序有依赖关系，不能随意调换：
            ① upsert 指纹（必须先有 source_document 行，否则父块的外键插入会失败）
            ② 删除该文档的旧父块（子块由外键级联一并删除）
            ③ 插入新父块（此时子块的外键才有指向）
            ④ 插入新子块（带向量）
        整个序列在**同一个事务**里：要么全部生效，要么全部回滚。
        否则中途失败会留下"指纹已更新、内容还是旧的"这种不一致状态，
        下次索引会以为"文件没变"而跳过，导致索引永远停在半新半旧的状态。
        """
        import numpy as np
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                # ① upsert：用 ON CONFLICT 实现"有则更新、无则插入"，
                #    比"先查再决定 insert/update"少一次往返，也没有竞态窗口
                cur.execute("""
                    INSERT INTO source_document (source, fingerprint)
                    VALUES (%s, %s)
                    ON CONFLICT (source) DO UPDATE SET
                        fingerprint = EXCLUDED.fingerprint, indexed_at = now()
                """, (source, fingerprint))
                # ② 只删父块即可：child_chunk 的外键带 ON DELETE CASCADE
                cur.execute("DELETE FROM parent_chunk WHERE source = %s", (source,))
                # ③ 用 executemany 批量插入父块（列表元素是 dict，按 %(key)s 占位符取值）
                cur.executemany("""
                    INSERT INTO parent_chunk
                    (parent_id, parent_text, source, heading, parent_index, char_len)
                    VALUES (%(parent_id)s, %(parent_text)s, %(source)s, %(heading)s,
                            %(parent_index)s, %(char_len)s)
                """, parents)
                # ④ 插入子块。向量必须先转成 float32 数组：
                #    pgvector 期望固定的数值精度，Python 的 float64 列表会导致类型不匹配
                cur.executemany("""
                    INSERT INTO child_chunk
                    (child_id, parent_id, child_text, embedding, child_index, char_start, char_end)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, [
                    (child["child_id"], child["parent_id"], child["child_text"],
                     np.asarray(child["embedding"], dtype=np.float32), child["child_index"],
                     child["char_start"], child["char_end"])
                    for child in children
                ])
            conn.commit()                                # ★ 四步一起提交
        except Exception:
            conn.rollback()                              # ★ 任一步失败则整体回滚
            raise
        finally:
            self.pool.putconn(conn)

    def delete_document(self, source):
        """删除文档。只需删 source_document 一行——
        父块与子块由两级外键 ON DELETE CASCADE 自动清理。
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM source_document WHERE source = %s", (source,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def search_children(self, query_vec, top_k, min_sim=MIN_SIM):
        """用 pgvector 的 `<=>` 余弦距离算子做近似最近邻检索。

        两个必须理解的细节：

        1. `<=>` 返回的是**距离**（越小越近），而我们需要"相似度"（越大越相关），
           所以用 `1 - (embedding <=> %s)` 换算。这是最容易写错的地方。

        2. 算子必须与索引类型一致：HNSW 索引是用 `vector_cosine_ops` 建的，
           查询也必须用余弦算子才能命中该索引。若换成 `<->`（欧氏距离），
           PostgreSQL 会退化为全表扫描——结果还"对"，但慢很多且不易察觉。

        SET LOCAL hnsw.ef_search 是"每次查询"调节召回率/速度的旋钮：
        值越大召回越全但越慢；LOCAL 表示只对当前事务生效，不污染连接池里的其他使用者。
        """
        import numpy as np
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
                cur.execute("""
                    SELECT child_id, parent_id, child_text, child_index, char_start, char_end,
                           1 - (embedding <=> %s) AS score
                    FROM child_chunk
                    WHERE 1 - (embedding <=> %s) >= %s
                    ORDER BY embedding <=> %s LIMIT %s
                """, (np.asarray(query_vec, dtype=np.float32), np.asarray(query_vec, dtype=np.float32),
                      min_sim, np.asarray(query_vec, dtype=np.float32), top_k))
                # 从游标描述里取列名，再和每行数据 zip 成 dict。
                # 这样列名只在 SQL 里写一次，不需要在 Python 里重复维护一份字段列表。
                columns = [item[0] for item in cur.description]
                # ★ 与 MemoryStore 保持一致：score 是 `1 - 余弦距离`，即**原始余弦相似度**，
                #   必须带上量纲标记，下游的拒答门禁才敢拿它和绝对阈值比较。
                return [dict(zip(columns, row), **{SCORE_KIND_KEY: SCORE_COSINE})
                        for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def list_children(self):
        # ⚠ 一次性 SELECT 全表：BM25 需要全量语料算 IDF，所以这里没有分页。
        #   语料很大时会占内存，改进方向见改进方案 P1-6（改用服务端全文索引）。
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT child_id, parent_id, child_text, child_index, char_start, char_end FROM child_chunk")
                columns = [item[0] for item in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        finally:
            self.pool.putconn(conn)

    def get_child_vectors(self, child_ids):
        # 空列表直接返回：先拦住可以避免发一条 `= ANY(ARRAY[])` 的无意义查询
        if not child_ids:
            return {}
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                # ANY(%s) 一次查多个 ID（批量），避免 N+1 次查询
                cur.execute("SELECT child_id, embedding FROM child_chunk WHERE child_id = ANY(%s)", (list(child_ids),))
                # str(row[0])：数据库返回的 UUID 是 uuid 对象，而调用方用的是字符串 ID，
                # 统一转成字符串才能作为 dict 键匹配上
                return {str(row[0]): list(row[1]) for row in cur.fetchall()}
        finally:
            self.pool.putconn(conn)

    def get_parents(self, parent_ids):
        if not parent_ids:
            return {}
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT parent_id, parent_text, source, heading, parent_index, char_len
                    FROM parent_chunk WHERE parent_id = ANY(%s)
                """, (list(parent_ids),))
                columns = [item[0] for item in cur.description]
                # 以 parent_id 为键返回 dict，方便调用方 parents.get(hit["parent_id"]) 直接取用
                return {str(row[0]): dict(zip(columns, row)) for row in cur.fetchall()}
        finally:
            self.pool.putconn(conn)
