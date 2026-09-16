# -*- coding: utf-8 -*-
"""② 切分层：Markdown 结构感知 + 两级父子块。

--------------------------------------------------------------------------------
这个模块要解决的核心问题
--------------------------------------------------------------------------------
同一个文档，需要两种**互相矛盾**的粒度：

  · 检索要"细"：块越小，向量表示越聚焦，越容易和查询对上 → 召回率高
  · 生成要"粗"：块越大，上下文越完整，LLM 越不容易看到半句话 → 答案质量高

一刀切的切分必然牺牲一边。本模块的做法是**同时产出两种粒度**：
    child（≤300 字符）—— 唯一参与 embedding，负责"被搜到"
    parent（≤1200 字符）—— 负责"被读懂"，检索命中子块后再回表取父块
两者用 parent_id 关联（1:N），所以叫"两级父子块"。

--------------------------------------------------------------------------------
为什么不能简单地按固定字数切
--------------------------------------------------------------------------------
四类内容切坏了都会造成不可逆的损伤：
    1. 代码块 —— 从中间切断后，语法都不成立，LLM 无法理解，检索也失去意义
    2. 标题   —— 标题被切到上一个块里，后面的正文就失去了"我在讲什么"的线索
    3. 段落   —— 一个论点被拆到两块，两块各自都不完整
    4. 句子   —— 最次的选择，但比字符硬切好

因此本模块的处理顺序是"逐级降级"：
    结构边界（标题/段落/代码块）  →  句子边界  →  字符硬切
能用高优先级边界就在那一级解决，实在不行才降级。

--------------------------------------------------------------------------------
Reader's guide（建议按这个顺序读代码）
--------------------------------------------------------------------------------
    1. MarkdownSplitter.split()   —— 状态机，按结构切出"块"，并记住每个块的标题路径
    2. MarkdownSplitter.hard_split() —— 块仍然太长时的降级切法
    3. build_parent_child()       —— 把块规范成"父块 + 滑窗子块"的 1:N 结构
"""
import hashlib
import re
import uuid
from typing import Dict, List, Tuple

from .config import (
    CHILD_OVERLAP,
    FENCE,
    MAX_CHILD_CHARS,
    MAX_PARENT_CHARS,
    MERGE_RATIO,
    MIN_CHILD_CHARS,
    SENT_END,
)


def chunk_signature(max_parent: int = MAX_PARENT_CHARS,
                    max_child: int = MAX_CHILD_CHARS,
                    overlap: int = CHILD_OVERLAP,
                    min_child: int = MIN_CHILD_CHARS,
                    merge_ratio: float = MERGE_RATIO) -> str:
    """把"决定切分结果的全部参数"压成一个短签名（8 位十六进制）。

    ★ 为什么必须有这个东西（本模块最容易被自己骗过去的一个坑）：
      增量索引原先只比较**文件内容的哈希**。但同一个文件切出什么块，除了内容
      还取决于上面这些参数——把 MAX_CHILD_CHARS 从 300 改成 150 之后重跑索引，
      文件内容没变 → 每个文件都被判定为"未变动"而跳过 → **新参数完全没生效**，
      而统计里显示的却是漂亮的 "skipped=N"（看起来还很"高效"）。

      签名进指纹之后，参数一变签名就变，"跳过"立刻变成"重建"，
      而且日志能说清是哪一种变化（内容变了 / 参数变了 / 模型变了）。

    为什么把参数显式列在签名里、而不是读一次 config 就算：调用方可以传非默认值
    （build_parent_child 支持覆盖），签名必须反映**实际生效的那一组**，
    否则"传了自定义尺寸"这种用法又会漏判。
    """
    raw = f"parent={max_parent}|child={max_child}|overlap={overlap}|" \
          f"min_child={min_child}|merge={merge_ratio}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


class MarkdownSplitter:
    """结构感知切分：保护代码块 + 提取标题面包屑。"""

    @staticmethod
    def split(text: str) -> List[Tuple[str, str]]:
        """把整篇文档切成 [(标题路径, 块文本)]。

        这是本模块最需要理解的一段，它是一个**单遍扫描的状态机**：
        逐行读入，根据"当前处于什么状态"决定这行该归到哪里。

        三个状态变量各司其职：
            buf     —— 正在累积的"当前块"。攒够一个段落/代码块就 flush 出去
            in_code —— 是否位于 ``` 代码围栏内部。**它决定空行和 # 是否还有特殊含义**
            heads   —— 当前各层级标题，形如 {1: "第三章", 2: "配置示例"}
                       用来生成"标题面包屑"，让每个块都知道自己属于哪一节

        为什么要用"累积 + flush"而不是直接 append 每行：
           因为一个块由多行组成，只有读到"块结束信号"（空行 / 新标题 / 围栏闭合）
           才知道这个块完整了。flush() 就是把 buf 里的内容固化成一个块并清空。

        块结束信号有三种，对应代码里三处 flush：
            ① 遇到空行         → 段落边界（Markdown 里空行就是段落分隔）
            ② 遇到新的标题行   → 小节边界（标题本身不进块，只更新 heads）
            ③ 代码围栏闭合     → 代码块边界（整段代码必须待在一个块里）
        """
        blocks: List[Tuple[str, str]] = []
        buf: List[str] = []
        in_code = False
        heads: Dict[int, str] = {}

        def flush():
            """把 buf 里累积的内容固化成一个块，然后用标题路径标注它。

            注意 `sorted(heads)`：靠 key（标题层级）排序还原层级顺序，
            于是 {1:"第三章", 2:"配置示例"} 变成 "第三章 > 配置示例"。
            这就是"面包屑"——它既给 LLM 提供上下文，也给人工核对提供定位线索。
            """
            if buf:                                   # 空 buf 不产出空块
                path = " > ".join(heads[k] for k in sorted(heads))
                blocks.append((path, "\n".join(buf).strip()))
                buf.clear()                           # ★ 必须清空，否则内容会重复进入下一个块

        for line in text.splitlines():
            # ---------- 情形 1：代码围栏行 ----------
            # 围栏行本身既不是内容也不是标题，而是"状态切换开关"，
            # 所以它单独处理并 continue，不参与下面的空行/标题判断。
            if line.lstrip().startswith(FENCE):        # 围栏行
                if in_code:
                    # 闭合围栏：此刻整个代码块已经完整，把它作为一个块落盘。
                    # 若此时不 flush，代码会和**后面**的正文混进同一个块，
                    # 导致检索到"半段代码 + 半段说明"这种没有意义的内容。
                    buf.append(line)
                    flush()                            # 闭合 → 整块代码打包
                else:
                    # 开启围栏：先把围栏之前的正文落盘。
                    # 若不 flush，代码会和**前面**的段落黏在一起成为同一个块。
                    flush()                            # 开启 → 先落盘前文
                    buf.append(line)
                in_code = not in_code
                continue

            # ---------- 情形 2：围栏内部 ----------
            if in_code:
                # 代码块内部的一切都是字面量：空行保留（可能是有意的排版），
                # 以 # 开头的行也必须原样保留（那是注释/宏，不是标题）。
                # 这一步就是"保护代码块"的全部秘密——不改写、不判断、只累积。
                buf.append(line)
                continue

            # ---------- 情形 3：标题行 ----------
            m = re.match(r'^(#{1,6})\s+(.*)', line)    # 标题行
            if m:
                # flush 在前：标题属于下一个块，不能滞留进上一个块
                flush()
                lvl, title = len(m.group(1)), m.group(2).strip()
                # ★ 丢弃层级 >= 当前标题的旧标题。
                #   例如当前是 "## 配置示例"，又来了一个 "## 常见错误"：
                #   两者同级，后者不是前者的子节，所以要把 "配置示例" 删掉，
                #   否则面包屑会错误地变成 "配置示例 > 常见错误"（假父子关系）。
                heads = {k: v for k, v in heads.items() if k < lvl}
                heads[lvl] = title
                # 标题行本身不塞进 buf：它的信息已经通过 heads 传递给后续所有块了。
                continue

            # ---------- 情形 4：空行 = 段落边界 ----------
            if not line.strip():
                flush()
                continue

            # ---------- 情形 5：普通正文行 ----------
            # 只累积，不判断——连续多行正文会一直攒在同一个 buf 里，
            # 直到遇到空行/标题/围栏才落盘，这样"一个段落"自然成为一个块。
            buf.append(line)
        flush()                                       # ★ 循环结束后必须补一次 flush
        return blocks

    @staticmethod
    def hard_split(block: str, size: int) -> List[str]:
        """块仍然超过 size 时的降级切分：先按句子切，仍超长才按字符硬切。

        为什么需要它：上面的 split() 只保证"在结构边界处切"，
        但一个没有空行的超长段落、或一个几百行的代码块，仍然会超出上限。
        父块有字符上限（否则喂给 LLM 会超预算），所以必须再切一次。

        两级策略的排序依据是"语义损失从小到大"：
            句子切   —— 句子是完整语义单元，读者/模型都不受影响
            字符切   —— 只保证不超限，语义必然受损，是最后的兜底
        """
        if len(block) <= size:
            return [block]                            # 已经够短，原样返回

        # ---------- 第一级：按句子边界累积 ----------
        # re.split 用"后顾断言"在句末标点之后切分，所以标点会留在前一句里（不会被丢掉）。
        # 得到的是一个个句子片段，下面按 size 把连续句子"装箱"成若干块。
        parts, cur = [], ""
        for seg in re.split(SENT_END, block):
            seg = seg or ""
            if len(cur) + len(seg) <= size:
                # 装得下就继续累积——这样多个短句会合成一个块，
                # 避免产出大量只有一句话的碎片块（碎片块会拉低检索质量）
                cur += seg
            else:
                if cur:
                    parts.append(cur)                 # 装不下：先把手上的落盘
                cur = seg                            # 然后以当前句重新开箱
        if cur:
            parts.append(cur)                         # ★ 循环末尾补一次（最后一句还没落盘）

        # ---------- 第二级：仍然超长的片段按字符硬切 ----------
        # 走到这一步说明存在"单个句子就超过 size"的情况（长代码行、无标点长串）。
        # 没有语义边界可用，只能等长切分，保证"任何块都不超过 size"这一硬约束。
        out = []
        for p in parts:
            out.extend([p[i:i+size] for i in range(0, len(p), size)]
                       if len(p) > size else [p])
        # 切完会残留空白片段（例如换行、缩进），统一 strip 并丢掉空串
        return [x for x in (s.strip() for s in out) if x]


def build_parent_child(text: str, source: str,
                       max_parent: int = MAX_PARENT_CHARS,
                       max_child: int = MAX_CHILD_CHARS,
                       overlap: int = CHILD_OVERLAP,
                       source_label: str = "") -> Dict:
    """文档 → 父块 + 子块，规范化 1:N 结构。

    整个函数分三步，每步对应一个明确目的：

        第一步（收集 raw）：把结构块整理成"不超过 max_parent 的父块文本"
        第二步（生成 parent）：给每个父块算一个稳定 ID，并登记元数据
        第三步（生成 child）：在父块内部用滑窗切出子块，形成 1:N 关系

    "稳定 ID" 是这里的关键设计：ID 由**内容**派生（而不是随机数或自增号），
    于是「同样的文档重复索引 → 得到同样的 ID」，
    上层就能用"先删后插"实现幂等 upsert，不需要额外的去重逻辑，
    也不需要分布式锁（多进程同时索引同一文件不会产生重复行）。
    """
    parents: Dict[str, Dict] = {}
    children: List[Dict] = []
    seen_child = set()

    # ================= 第一步：把结构块整理成父块文本 =================
    raw: List[Tuple[str, str]] = []
    # ⚠ 循环变量刻意取名为 heading_path 而不是 path：
    #   函数的第一个参数就叫 `path`（文档路径），同名会在循环开始后把参数覆盖掉——
    #   这是一种"当时能跑、但如果之后要在循环内用到原参数就会出错"的隐患，
    #   同时也会让静态检查与读者都误判类型。
    #
    # ★ 是否把 source_label 作为**根节点**拼在面包屑前面：
    #   规则是"文档自身的一级标题用的是 `#` 还是更深"。
    #   · 原生 Markdown 用它自己的 `#` 作为顶层 → 不需要再套一层文件名（保持原行为）
    #   · 由 TXT 等转换来的文本，标题被统一降为 `##`（把 `#` 留给文件名）→ 补上根节点
    #   这样 `接口文档 > 第一章 概述` 才完整：既有小节层级，也保住了**文件身份**。
    #   否则检索结果只显示 "第一章 概述"，无法知道它来自哪个文件。
    top_heading_levels = re.findall(r"^(#{1,6})\s+\S", text, flags=re.MULTILINE)
    use_label_as_root = bool(source_label) and (
        not top_heading_levels or min(len(h) for h in top_heading_levels) > 1)

    for heading_path, block in MarkdownSplitter.split(text):
        # 面包屑兜底：文档自身没有任何标题时（纯文本、去标签后的 HTML），
        # 用 `source_label`（文件名主干）当根节点，否则这些块的 heading 会是空的，
        # 检索结果无法溯源到任何层级。
        if not heading_path and source_label:
            heading_path = source_label
        elif heading_path and use_label_as_root:
            heading_path = f"{source_label} > {heading_path}"
        # ★ 预留标题前缀的长度余量。
        #   下面会拼成 "[第三章 > 配置示例] 正文…"，前缀有长度。
        #   如果先按 max_parent 切好再拼前缀，成品就会超出上限——
        #   这类"边界多算了 20 个字符"的错误在喂给 LLM 时才暴露（超预算报错）。
        #   所以在**切之前**就把前缀的额度扣掉。
        prefix = f"[{heading_path}] " if heading_path else ""
        budget = max(1, max_parent - len(prefix))
        for piece in MarkdownSplitter.hard_split(block, budget):
            if piece:
                # 前缀 + 正文一起存为父块文本：这样父块自身就带着"出处线索"，
                # 即使单独看一个父块，也知道它来自哪一节。
                raw.append((heading_path, prefix + piece))

    # ================= 第二步 + 第三步：生成父块与其子块 =================
    for p_idx, (path, parent_text) in enumerate(raw):
        # 父块 ID = uuid5(命名空间, "来源::序号::内容")
        #   · uuid5 是**确定性**哈希：同样输入永远得到同样输出（uuid4 是随机的，不能用）
        #   · 掺入 source 是为了让不同文件里相同的内容得到不同 ID（否则会互相覆盖）
        #   · 掺入内容是为了让"改过的块"得到新 ID，从而能被识别为需要重建
        parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{p_idx}::{parent_text}"))
        if parent_id in parents:
            # 同一文档内出现完全相同的块（如重复的模板段落）时只保留一份。
            # 注意：这里是"父块去重"，与后面的"子块去重"是两件不同的事。
            continue                                   # 父块去重
        parents[parent_id] = {
            "parent_id": parent_id, "parent_text": parent_text,
            "source": source, "heading": path,
            "parent_index": p_idx, "char_len": len(parent_text),
        }

        # ---------- 滑窗切子块 ----------
        # 目标：用固定宽度的窗口扫过整个父块，相邻窗口互相重叠一部分。
        # 为什么必须重叠：如果切点正好落在一个完整答案的中间，
        #   不重叠时"前半句"和"后半句"分属两个块，两个块都不足以回答该问题；
        #   有重叠时至少有一个块完整包含答案 → 该答案有机会被检索到。
        # 代价：重叠部分被重复索引，存储与 embedding 成本按 (1 + overlap/step) 增加。
        step = max_child - overlap
        if step <= 0:
            # 步长非正会让下面的 while 循环永不前进（死循环）。
            # 这种"参数组合本身无意义"的情况必须响亮报错，而不是静默降级。
            raise ValueError(f"overlap({overlap}) 必须 < max_child({max_child})")

        spans, start = [], 0
        while start < len(parent_text):
            # 窗口右端不越过父块末尾（最后一块可能比 max_child 短）
            end = min(start + max_child, len(parent_text))
            spans.append((start, end))
            if end >= len(parent_text):
                break                                 # 已覆盖到末尾，结束
            # 只前进 step（而不是 max_child）：这个"少走一点"就是重叠的来源
            start += step

        # ---------- 合并过短的尾块 ----------
        # 滑窗很容易留下一个很短的尾块（比如只剩 5 个字）。
        # 这种碎片块的向量表示没有信息量，几乎不可能被检索到，纯属噪声。
        # 因此把它并进前一个块——但合并后不能超得太离谱，
        # 所以用 MERGE_RATIO 给一个"允许超限的倍率"（见 config.py 的说明）。
        if len(spans) > 1:                             # 合并短尾块且不超限
            ts, te = spans[-1]
            ps, _ = spans[-2]
            if te - ts < MIN_CHILD_CHARS and (te - ps) <= max_child * MERGE_RATIO:
                spans.pop()
                spans[-1] = (ps, te)

        for i, (s, e) in enumerate(spans):
            # 用切片取出子块文本：注意这是父块的**逐字切片**（不重写、不改写）。
            # 这一点很重要——评测的锚点、答案的原文引用都依赖"子块文本能在父块里逐字找到"。
            child_text = parent_text[s:e].strip()
            if not child_text:
                continue                              # 纯空白切片（如整块缩进）直接跳过
            # 子块 ID 同样由内容派生，且掺入 parent_id 与序号 i：
            # 保证"同一父块内的不同子块"有不同的 ID（否则重叠内容会撞 ID）。
            child_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                      f"{parent_id}::{i}::{child_text}"))
            if child_id in seen_child:
                # 重叠窗口有时会切出完全相同的文本（例如父块本身很短、或内容高度重复），
                # 这里去重，避免同一个向量被存多次、检索结果里出现重复条目。
                continue
            seen_child.add(child_id)
            children.append({
                "child_id": child_id, "parent_id": parent_id,
                "child_text": child_text,          # 唯一参与 embedding 的文本
                # 记录原始偏移：既能溯源（"这句话在父块的哪个位置"），
                # 也便于调试切片是否正确
                "child_index": i, "char_start": s, "char_end": e,
            })
    return {"parents": parents, "children": children}
