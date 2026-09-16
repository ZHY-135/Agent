# RAG Demo —— 带可视化观测台的检索增强生成流水线

> 一个**可直接运行**的 RAG（检索增强生成）示例项目：把「文档 → 切分 → 向量化 → 混合检索 → 重排 → 上下文 → 生成」整条链路完整实现，
> 并配一个**可视化观测台**——一次提问在每一环的中间结果都能看见（召回了几条、谁被淘汰、预算花在哪、为什么拒答）。
>
> 核心路径**零第三方依赖**：不装任何包也能跑通 `demo` / `qa` 与观测台之外的完整链路。

![CI](https://github.com/ZHY-135/Agent/actions/workflows/ci.yml/badge.svg)

![观测台截图](images/observatory.png)

---

## 📦 这个仓库里有什么

| 包含 | 说明 |
|---|---|
| **`RAG/`** | 全部代码，**20 个模块**（九环流水线 + 可视化观测台 + 工具脚本） |
| `pyproject.toml` / `uv.lock` | 打包声明与依赖精确锁定（`uv.lock` 保证别人装出完全一样的环境） |
| `.github/workflows/ci.yml` | CI：跨平台冒烟（Linux 3.10/3.11/3.12 + Windows）+ 静态检查 + 安装验证 |
| `README.md` / `LICENSE` / `.env.example` / `.gitignore` | 项目说明、MIT 许可证、环境变量模板、忽略规则 |

**不包含**（保留在开发机本地，`.gitignore` 里已逐条列明）：单元测试、评测集、内部设计文档与辅助脚本。
所以克隆这个仓库后**开箱即可运行**，但不附带开发期的回归测试与评测资产。

---

## ✨ 项目特点

| | |
|---|---|
| 🖥 **可视化观测台** | 浏览器里看清一次提问的全过程：8 阶段耗时瀑布、三路榜单并排、预算占用、拒答原因、引用溯源 |
| 📦 **开箱即用** | `uv sync` → `python -m RAG.main serve`，两分钟看到效果；无需 API Key、无需数据库 |
| 🧩 **每一层都可替换** | 换嵌入模型 / 换向量库 / 换 LLM，只改 `main.py` 装配处一行，编排层与业务代码零改动 |
| 🔍 **防幻觉三道防线** | 生成前拒答门禁 → 提示词强制引用 → 生成后 `[n]` 回填校验 |
| 📄 **多格式接入** | 统一转成 Markdown 再切分（TXT 完整实现；PDF / Word 留接口） |

---

## 🖥 可视化观测台

```bash
python -m RAG.main serve                  # 默认 http://127.0.0.1:8501，自动打开浏览器
python -m RAG.main serve --docs docs --port 8765
```

启动后输入问题，能看到五个标签页：

| 标签页 | 你能看到什么 |
|---|---|
| **① 总览** | 当前装配了哪些组件（嵌入器 / 向量库 / BM25 / MMR / 生成器）· 语料台账（每个文件是否需要重建**以及原因**）· 当前生效的切分参数 · 一键重建索引 |
| **② 检索瀑布** | 一次查询在 8 个阶段的**条数与耗时**（嵌入 → 向量召回 → 关键词召回 → RRF 融合 → 父块聚合 → MMR → 上下文装填 → 支持度）；**向量榜 / BM25 榜 / 融合榜三列并排**，看清每条是谁捞上来的 |
| **③ 上下文与 prompt** | token 预算进度条 · 每块占用的 token 柱状图 · 逐块正文 · 被预算挤掉的块 · 最终 prompt 原文 |
| **④ 生成与校验** | 答案 · 拒答原因码 · 引用校验五指标（引用数 / 合法 / 凭空编号 / 未使用资料 / 覆盖率）· 每条引用对应的原文出处 · 支持度与阈值对照 |
| **⑤ 评测** | 一键跑四组配置的检索指标表与柱状对比 |

侧边栏还能**实时切换** `use_bm25` / `use_mmr` / `top_k` / `min_sim`——取消勾选「启用 BM25」再回②，就能直接看到纯向量与混合检索的差距。

> 观测台**不重算任何检索逻辑**：它显示的每一个数字都取自 `retrieve()` / `answer()` 的返回值与一个可选的 `trace` 观测钩子（默认关闭、零开销），因此不存在"界面和实际行为不一致"的可能。

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/ZHY-135/Agent.git
cd Agent

uv sync --extra webui            # 核心 + 观测台依赖（推荐：这就够跑完整个演示）
# 想同时用静态检查工具（ruff / mypy）：uv sync --extra dev --extra webui
```

> 只用核心功能（不装 streamlit）也可以：`uv sync` 即可，`demo` / `qa` 照常可用。
> 用 pip 也行：`pip install -e ".[webui]"`。
> ⚠️ `uv sync` 会让环境精确匹配 `uv.lock` + 所选 extra：不带 `--extra webui` 会把 streamlit 清掉。

### 2. 启动可视化

```bash
python -m RAG.main serve
```

浏览器打开 `http://127.0.0.1:8501`，输入问题即可看到全链路。

### 3. 命令行跑一遍

```bash
python -m RAG.main demo                        # 建索引 + 打印最终 prompt（零依赖，无需 API Key）
python -m RAG.main qa "连接池最大连接数是多少"   # 答案 + 引用出处 + 校验结果
python -m RAG.debug_trace                      # 按 9 个阶段打印中间结果（文本版剖面）
```

> 💡 **第一次运行会自动生成示例语料**：仓库里没有预置文档，`demo` / `qa` / `serve` 会在 `docs/` 下
> 自动写一份演示文档（`demo_rag_doc.md`）并建索引，所以可以立刻看到效果。想用自己的文档见下一步。

### 4. 换成你自己的文档

```bash
python -m RAG.main index  /path/to/你的文档目录
python -m RAG.main qa "你的问题" --docs /path/to/你的文档目录
python -m RAG.main qa --chat --docs /path/to/你的文档目录   # 多轮追问（拼历史，不做改写）
python -m RAG.main serve --docs /path/to/你的文档目录        # 面板里也能随时改
```

面板左侧「📁 语料目录」还提供：**系统原生「选择文件夹」对话框**、路径实时校验、常用目录收藏、自动记住上次选择。

支持格式（下表与代码实际行为一致）：

| 格式 | 状态 |
|---|---|
| `.md` / `.markdown` | ✅ 完整支持（本身就是中间表示） |
| `.txt` | ✅ 完整支持：编码兜底（UTF-8 → GB18030）、标题识别、硬换行段落重排 |
| `.html` / `.htm` | ✅ **保留标题层级**（`h1`–`h6` → `##`…，代码块加围栏，导航/页脚/脚本被剔除）。未装 `beautifulsoup4` 时自动**降级**为正则去标签——**降级仍会进索引**，只是面包屑退化为文件名 |
| `.docx` | ✅ 支持：保留标题层级（含中文样式名「标题 1」）、列表，且**表格按文档顺序**插入。需 `python-docx`（`pip install "rag-min[loaders]"`）；未装时会被跳过并在索引汇总与台账里标注原因 |
| `.pdf` | ❌ **不支持直接索引**（刻意不做，见下文「已知限制」）。会被扫描到、被点名，并给出**离线转换指引**（docling / pymupdf4llm / markitdown 任选其一） |
| `.doc` | ❌ 不支持（Word 97-2003 二进制格式）。会被扫描到并提示「另存为 .docx」 |

### 5. 用真实语义向量（可选，两种方式）

**方式 A：本机模型，不需要 API Key**（推荐先试这个）

```bash
pip install "rag-min[local]"              # fastembed / ONNX，几十 MB（不是 PyTorch）
export RAG_EMBEDDING_PROVIDER=local
python -m RAG.main qa "连接池上限是多少" --docs ./你的语料目录
```

默认模型是 `BAAI/bge-small-zh-v1.5`（512 维，中文小模型）；可用 `RAG_LOCAL_MODEL` 换模型，
可选清单见 `RAG/config.py` 的 `LOCAL_EMBED_MODELS`。
⚠ **首次使用会联网下载模型权重**（约 90 MB，之后走本地缓存）；完全离线请继续用默认的伪向量。

> 说明：本项目的检索质量**目前还没有在真实评测集上量化过**（评测集需要人工标注 `reviewed`）。
> 所以这里只说"能用本机模型做语义检索"，**不声称它比 `openai` 或比 hash 好多少**——
> 评测集跑完 recall@k / nDCG 之后，能力对比才有据可依。

**方式 B：OpenAI，需要 API Key**

```bash
export RAG_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export RAG_BACKEND=pgvector               # 用 PostgreSQL + pgvector 持久化（默认是内存）
export DATABASE_URL=postgresql://user:pass@localhost:5432/rag
python -m RAG.main init-db                # 建表与 HNSW 索引
```

> ⚠ **换嵌入模型后请重新标定 `MIN_SIM` / `REFUSE_MIN_VEC_SCORE`**：这两个阈值（默认 0.35）
> 是按 `text-embedding-3-small` 的余弦分布标定的，换模型后分布会变——沿用旧阈值可能
> 表现为"该答的拒答了"或"不该答的放过去了"。
>
> ✅ 但**索引会自动重建**：嵌入器身份已计入"索引签名"，`index()` 会发现签名变化并整篇重算，
> 不需要你手动清库。（详见下文"一致性与可靠性防护"）

---

## 🧩 实现了哪些功能

### 文档接入
- 目录递归扫描、后缀白名单、编码兜底（UTF-8 → GB18030）、BOM 剥离
- **内容指纹增量索引**：文件没变就整篇跳过，不重复转换、不重复切分、不重复算向量
- **目录同步**：磁盘上已删除的文件，其索引会被自动清理
- **单文件也校验后缀**：传 `.xlsx` / `.py` 会当场报错并给出修复指引，而不是读成乱码静默入库
- 统一转 Markdown 的转换层：TXT 完整实现（硬换行重排、保守标题识别、文件名兜底面包屑），PDF / Word 留接口

### 切分
- Markdown 结构感知切分：代码围栏不被切断、标题与正文不分离
- **父子两级块**：子块 ≤300 字（用于向量匹配）+ 父块 ≤1200 字（用于喂给模型）
- 滑窗重叠（50 字）、过短尾块自动合并、超长块按"段落 → 句子 → 字符"逐级降级
- **标题面包屑**（如 `第三章 > 3.2 常见错误`）；无标题的格式用文件名兜底，保证结果可溯源
- 内容派生 UUID：同样内容重复索引得到同样的 ID（幂等）

### 嵌入
- 批量调用（64/批）、指数退避重试、维度校验、按内容哈希缓存
- 三种实现：`HashEmbedder`（零依赖伪向量，离线演示用）/ `LocalEmbedder`（本机真实语义向量，**免 API Key**）/ `OpenAIEmbedder`（生产用）

### 存储
- `VectorStore` 抽象 + 两种实现：`MemoryStore`（内存，演示）/ `PgVectorStore`（PostgreSQL + pgvector）
- pgvector 侧：三表外键级联、HNSW 向量索引、事务化"先删后插"实现幂等更新

### 检索
- **向量通道**（语义相近）+ **BM25 关键词通道**（字面精确，专治错误码 / API 名）
- **RRF 融合**：只看排名不看分数，避开两侧量纲不可比的问题
- **先按父块聚合、后截断**：保证返回的几条来自不同章节，而不是同一节的碎片
- 三个通道的原始结果都对外暴露（向量榜 / 关键词榜 / 融合后），便于观察与调试

### 重排与上下文
- MMR 多样性重排：去冗余，避免把高度重复的块塞满上下文
- **token 预算贪心装填**：整块放得下就放整块，放不下退化为命中片段，再放不下则记录为"被丢弃"并告警
- 参考资料统一编号 `[1][2]…`，供生成层强制引用与人工溯源

### 生成与防幻觉
- **三层防线**：生成前拒答门禁 → 严格系统提示词（只依据资料、强制标注 `[n]`）→ 生成后引用回填校验
- 明确的拒答原因码：无资料 / 支持度过低 / 引用了不存在的编号 / 答案无任何合法引用
- 拒答话术由代码固化，不让模型自由发挥
- 两种生成器：`EchoGenerator`（离线复述资料，用于跑通全链路）/ `OpenAIGenerator`

### 可视化观测台
- 5 个标签页（见上文），覆盖从召回到拒答的每一个可观测环节
- 侧边栏实时切换检索开关与参数，改完自动重建索引
- 语料台账会直接告诉你"这个文件为什么需要重建"（内容变了 / 切分参数变了 / 换了嵌入模型）

### 语料目录管理
- **系统原生「选择文件夹」对话框**（tkinter 标准库，跑在独立子进程，不阻塞服务线程）
- 目录路径实时校验 + 自动记住上次选择 + 收藏常用目录

### 一致性与可靠性防护

| 机制 | 作用 |
|---|---|
| **索引签名** | 指纹 = 文件内容哈希 + 切分参数 + 嵌入模型标识；改了参数或换了模型会真的重建，不会用旧索引假装成功 |
| **分数量纲标记** | 每条命中标注自己是余弦分 / BM25 分 / RRF 融合分；把不可比的分数混用会直接报错，而不是算出一个错结果 |
| **路径与后缀校验** | 非法后缀当场报错；盘符根（如 `D:\`）不会被错误清洗 |

---

## 🏗 架构

### 流水线九环

| 环 | 环节 | 作用 | 代码 |
|---|---|---|---|
| ① | 加载 | 找到文件、读出内容、算内容指纹 | `loaders.py` |
| ①.5 | 转换 | 各格式统一转 Markdown | `converters.py` |
| ② | 切分 | 结构感知切分 + 父子两级块 | `chunkers.py` |
| ③ | 嵌入 | 文本 → 向量 | `embedders.py` |
| ④ | 存储 | 向量与块的持久化、检索 | `stores.py` |
| ⑤ | 检索 | 向量 + BM25 → RRF → 父块聚合 | `retrievers.py` |
| ⑥ | 重排 | MMR 多样性去冗 | `rerankers.py` |
| ⑦ | 上下文 | token 预算装填 + 引用编号 | `context.py` |
| ⑧ | 生成 | 调模型 + 引用校验 + 拒答 | `generators.py` |
| ⑨ | 编排 / 评测 | 串起全链路；把检索质量变成可复现的数字 | `pipeline.py` / `evaluation.py` |

> ①②③④ 是**索引期**（可增量），⑤⑥⑦⑧ 是**查询期**。

### 数据流

```
【索引期】
  文件 ─loaders──▶ 内容与指纹 ─converters──▶ Markdown
       ─chunkers──▶ 父块 + 子块 ─embedders──▶ 向量
       ─stores.replace_document()──▶ 入库（事务内先删后插）
       ─BM25.fit(全量子块)──▶ 关键词索引

【查询期】
  问题 ─embed──▶ 查询向量
       ├─ store.search_children()   向量召回
       └─ BM25.search()             关键词召回
              └─ rrf_fuse()   两路按排名融合
                    └─ aggregate_by_parent()   ★ 先聚合后截断
                          └─ mmr()             多样性去冗
                                └─ get_parents()   回表取父块全文
                                      └─ pack_context()  预算装填 + 编号
                                            └─ render_prompt() 或
                                               answer()：拒答门禁 → 生成 → 引用校验
```

### 分层与依赖方向

```
main.py（装配：唯一知道"用哪个具体实现"的地方）
   └─▶ pipeline.py（编排：index / retrieve / query / answer）
          └─▶ 功能层（loaders / converters / chunkers / embedders / stores /
                      retrievers / rerankers / context / generators）
                 └─▶ config.py（参数）+ settings.py（环境变量）
```

- `pipeline` 只依赖 `Embedder` / `VectorStore` / `Generator` 三个抽象，因此**换实现不改编排层**
- `query()` 与 `answer()` 共用同一条 `retrieve()` 路径，行为一致
- 无循环依赖

### 可替换点

| 换什么 | 改哪里 | 现有实现 |
|---|---|---|
| 嵌入模型 | `main.build_embedder()` | `HashEmbedder`（零依赖伪向量，默认）/ `LocalEmbedder`（本机真实语义向量，免 Key）/ `OpenAIEmbedder`（需 Key） |
| 向量库 | `main.build_store()` | `MemoryStore`（演示）/ `PgVectorStore`（生产） |
| LLM | `main.build_generator()` | `EchoGenerator`（离线）/ `OpenAIGenerator` |
| 切分与检索参数 | `config.py` | 一个文件管全部可调参数 |

---

## 📁 目录结构与模块

```
<仓库根>
├── RAG/                          ← Python 包（20 个模块）
│   ├── webui.py                  ← 可视化观测台（Streamlit，5 个标签页）
│   ├── pipeline.py               ← 编排层
│   ├── corpus.py                 ← 语料目录管理
│   └── …                         ← 其余模块（见下表）
├── .github/workflows/ci.yml      ← CI：跨平台冒烟 + 静态检查 + 安装验证
├── pyproject.toml                ← 打包与依赖声明（readme 指向本文件）
├── uv.lock                       ← 依赖精确锁定（建议一并提交/下载）
├── README.md                     ← 本文件
└── LICENSE / .env.example / .gitignore
```

| 模块 | 职责 |
|---|---|
| `pipeline.py` | **编排层**：`index()` / `retrieve()` / `query()` / `answer()`，以及索引签名与增量判据 |
| `main.py` | CLI 入口（`demo` / `qa` / `index` / `query` / `eval` / `serve`）与唯一的依赖装配处 |
| `config.py` / `settings.py` | 可调参数集中管理；环境变量与密钥（密钥不入库） |
| `loaders.py` | ① 加载：目录递归、编码兜底、内容指纹、单文件后缀校验 |
| `converters.py` | ①.5 转换：txt / html / pdf / docx → Markdown |
| `chunkers.py` | ② 切分：Markdown 状态机 + 父子两级块 + 面包屑 |
| `embedders.py` | ③ 嵌入：批量 / 重试 / 缓存 / 维度校验 |
| `stores.py` | ④ 存储：抽象 + 内存实现 + pgvector 实现 |
| `retrievers.py` | ⑤ 检索：BM25 + RRF 融合 + 父块聚合 |
| `rerankers.py` | ⑥ 重排：MMR（另留 CrossEncoder 接口） |
| `context.py` | ⑦ 上下文：token 预算装填 + 引用编号 |
| `generators.py` | ⑧ 生成：模型调用 + 引用回填校验 + 拒答策略 |
| `evaluation.py` | ⑨ 评测：锚点匹配 + 指标计算 + 报告 |
| `webui.py` | **可视化观测台**（Streamlit，5 个标签页） |
| `corpus.py` | 语料目录管理：路径校验、系统对话框、设置持久化、收藏夹 |
| `debug_trace.py` | 逐步调试：按 9 个阶段打印中间结果 |
| `stress_test.py` / `logging_setup.py` | 预算压测脚本；日志基础设施 |

---

## ⚙️ 常用配置（`RAG/config.py`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `MAX_CHILD_CHARS` / `MAX_PARENT_CHARS` | 300 / 1200 | 子块（参与向量匹配）/ 父块（喂给模型）的字符上限 |
| `CHILD_OVERLAP` | 50 | 子块滑窗重叠，避免答案被切断 |
| `TOP_K` | 5 | 最终返回条数 |
| `CANDIDATE_MUL` | 4 | 先召回 `top_k × 4` 再精排 |
| `MMR_LAMBDA` | 0.7 | 相关性 vs 多样性的权衡 |
| `CONTEXT_BUDGET` | 6000 | prompt 的 token 预算 |
| `REFUSE_MIN_VEC_SCORE` | 0.35 | 支持度低于此值则拒答（与**原始余弦**比较） |

环境变量：`RAG_BACKEND`（memory / pgvector）、`RAG_EMBEDDING_PROVIDER`（hash / local / openai）、`RAG_EMBED_MODEL`、`RAG_LOCAL_MODEL`、`RAG_EMBED_DIM`、`DATABASE_URL`、`OPENAI_API_KEY`。

---

## ⚠️ 已知限制

- **默认是伪向量**（`HashEmbedder`）：只认得字面重合，用于零依赖演示。要真正的语义检索有两条路：`RAG_EMBEDDING_PROVIDER=local`（**本机模型，免 API Key**，装 `rag-min[local]`）或 `openai`（需 Key）。⚠ **换模型后请重新标定 `MIN_SIM` / `REFUSE_MIN_VEC_SCORE`**——这两个阈值是按 `text-embedding-3-small` 的余弦分布标定的。
- **`local` 模式首次运行需联网下载模型权重**（bge-small-zh 约 100 MB），之后走本地缓存。完全离线的环境请继续用默认的 hash：它零依赖、能跑通全链路，只是检索质量**仅供回归对比**，不能当效果结论。
- **PDF 不支持直接索引（刻意不做）**：PDF 只有版式坐标、**没有可靠的语义层级**，而标题正是本项目面包屑与溯源的骨架——公开基准里最好的 docling 标题识别也只有 0.824，而**错误的面包屑比没有面包屑更糟**（会让检索结果张冠李戴）。放进语料目录的 `.pdf` 会被扫描到，并在索引汇总与观测台台账里**点名说明"不参与索引"**，同时给出离线转换指引（docling / pymupdf4llm / markitdown 任选其一，先转 Markdown 再入库）。
- **`.doc` 不支持**：Word 97-2003 是老二进制格式，没有公开稳定的结构，请先在 Word / WPS 里「另存为」`.docx`。
- **DOCX 需要 `python-docx`**：`pip install "rag-min[loaders]"`。未装时 `.docx` 会被跳过，并在索引汇总与台账里标注「缺解析依赖」——**不会静默忽略**。
- **HTML 未装 `beautifulsoup4` 时标题层级会丢**：会自动降级为正则去标签——内容仍然进索引（这一点是刻意的：少层级远好于搜不到），但面包屑退化为文件名。装 `pip install "rag-min[loaders]"` 即可保留结构（详见上文「支持格式」表）。
- **多轮追问只做"拼历史"，不做查询改写**（`rag qa --chat`，或面板勾选「多轮追问」）：历史会被拼进 prompt 并从资料预算里扣走（上限 `HISTORY_MAX_TOKENS`），且历史段**不带 `[n]` 编号**，以免被引用校验误认为资料。所以**追问里出现上一轮原词时**模型能借助上下文；但**纯指代（"它的默认值呢？"）仍会检索跑偏**——检索用的是原始问句，里面没有可检索的实词。实测这一问的向量支持度为 `0.0000`，只靠少量关键词命中凑出一个引用。后续改进方案是先用一次 LLM 调用把追问改写成独立查询（已列入后续计划）。
- **内存模式不跨进程持久化**：需要持久化请用 `RAG_BACKEND=pgvector`。
- **BM25 与内存检索是暴力扫描**：适合中小语料；更大规模建议交给数据库全文索引与 pgvector。
- **本仓库不含测试与评测资产**：`tests/`、`eval/` 等保留在开发机本地，因此 `eval` 子命令需要自备评测集才能使用。

---

## 📄 许可证

本项目采用 **MIT License**，可自由使用、修改、分发（包括商用），只需保留版权声明与许可证原文。
全文见 [`LICENSE`](LICENSE)。
