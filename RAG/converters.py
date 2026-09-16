# -*- coding: utf-8 -*-
"""① 文档转换层：把各种格式统一转成 **Markdown**，再交给切分层。

--------------------------------------------------------------------------------
为什么要统一转 Markdown（本模块存在的唯一理由）
--------------------------------------------------------------------------------
`chunkers.py` 只认 Markdown 语法：
    · 标题面包屑  依赖 `#` / `##` 前缀  → 非 Markdown 输入会拿不到目录结构，面包屑全空
    · 代码块保护  依赖 ``` 围栏         → 否则代码被当普通段落切碎
    · 段落边界    依赖空行             → PDF 的换行是排版换行，不是段落

所以**不给每种格式写一套切分逻辑**，而是全部转成 Markdown，复用同一条链路：

    .txt / .html / .docx → converters.convert() → Markdown → loaders → chunkers

好处：① 切分器一行都不用改 ② 转换器可独立测试（契约：路径 → Markdown）
      ③ 依赖隔离（HTML/Word 的可选库只在用到时才需要）

行业做法与此一致：MarkItDown(微软)、Docling(IBM)、MinerU、Marker、pymupdf4llm
全部输出 Markdown。

--------------------------------------------------------------------------------
当前实现状态（必须与 README 的"支持矩阵"保持一致）
--------------------------------------------------------------------------------
    ✅ convert_txt    —— 纯标准库，已完整实现
    ✅ convert_html   —— bs4 保留标题层级；**无 bs4 时降级为正则去标签**（不丢文件）
    ✅ convert_docx   —— 需要 python-docx（`rag-min[loaders]` 已含它）；保留标题、列表与表格
    ❌ convert_pdf    —— **刻意不做**：改为"明确拒绝 + 离线转换指引"（见 PDF_GUIDANCE）

--------------------------------------------------------------------------------
关于 `reason`：为什么 `available` 这一个布尔值不够用
--------------------------------------------------------------------------------
`available` 的原语义是"**依赖是否齐备**"——为 False 时调用方会跳过该文件
**且不登记指纹**（见 pipeline.index）。但现实中"跳过"有三种完全不同的原因：

    依赖缺失（装个包就能好） / 功能未实现（怎么装都没用） / 主动降级（能读，只是差些）

挤在同一个布尔值里，调用方与界面就无法给出不同提示——用户只会看到
"我放了文件却没反应"。因此新增 `reason` 字段与 `available` **并存**：
`available` 决定"能不能用"，`reason` 决定"为什么不能/为什么差"。

⚠ 最典型的历史 bug：HTML 明明能读，却被标成 `available=False`，
于是 `.html` 被整篇丢弃（台账显示"1 篇文档"，索引却是 0 块）。
**"降级"不等于"不可用"**，这是本模块最容易再犯的错误。
"""
import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .logging_setup import get_logger

logger = get_logger(__name__)

__all__ = ["ConvertResult", "convert", "convert_txt", "convert_html",
           "convert_docx", "convert_pdf", "is_supported", "index_capability",
           "SUPPORTED_KINDS", "strip_bom", "decode_text", "reflow_paragraphs",
           "looks_like_heading", "strip_html_tags", "PDF_GUIDANCE",
           "REASON_DEP_MISSING", "REASON_NOT_IMPLEMENTED", "REASON_DEGRADED",
           "REASON_PARSE_ERROR"]

# `ConvertResult.reason` 的取值。集中定义而不是散落字面量：
# 调用方（pipeline / corpus / webui）要按它分流提示，拼错字符串不会报错、只会静默失效。
REASON_DEP_MISSING = "dep_missing"          # 装个包就能好
REASON_NOT_IMPLEMENTED = "not_implemented"  # 怎么装都没用
REASON_DEGRADED = "degraded"                # 能读，但丢了一部分结构
# 文件本身有问题（损坏 / 加密 / 后缀伪装）：与"我们没做"是两回事，
# 用户能采取的行动也不同（重新导出 vs 换个格式），所以单独一个原因码。
REASON_PARSE_ERROR = "parse_error"

# 原因码 → 给用户看的**行动指引**。为什么要映射而不是直接打印原因码：
# 用户看到 "not_implemented" 不知道要做什么；而这两类原因的行动方向**完全相反**——
# 一个"去装包就行"，另一个"装什么都没用，得换个做法"。混在一起说等于没说。
# 放在这里而不是各自的界面文件里：CLI（main）、语料台账（corpus）、观测台（webui）
# 三处都要用同一句话，否则同一种问题会出现三种说法。
REASON_HINT = {
    REASON_DEP_MISSING: "缺解析依赖（装上对应的包即可，安装命令见日志）",
    REASON_NOT_IMPLEMENTED: "当前版本不支持该格式（请先转成 Markdown）",
    REASON_PARSE_ERROR: "文件损坏或格式不符（建议重新导出）",
    REASON_DEGRADED: "已降级索引（内容搜得到，但结构有损）",
}

# PDF 的离线转换指引。给出三条可选路径，用户拿任意一条都能先把 PDF 转成 Markdown。
PDF_GUIDANCE = (
    "PDF 暂不支持直接索引（本项目不内置 PDF 解析）。请先用离线工具转成 Markdown 再放入语料目录，"
    "可选：docling / pymupdf4llm / markitdown（任选其一，均输出 Markdown）。"
    "原因：PDF 只有版式坐标、没有可靠的语义层级，而标题正是本项目面包屑与溯源的骨架——"
    "错误的面包屑比没有面包屑更糟。"
)

# 后缀 → 文档类型。集中一处便于与 loaders.SUPPORTED 对齐
KIND_BY_SUFFIX: Dict[str, str] = {
    ".md": "markdown", ".markdown": "markdown",
    ".txt": "text",
    ".html": "html", ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    # `.doc` 登记进来的目的**不是**要支持它，而是让它"被看见"：
    # 不登记的话，语料目录里的 .doc 会被扫描跳过且毫无提示（见 _doc_unsupported）。
    ".doc": "doc",
}
SUPPORTED_KINDS = tuple(sorted(set(KIND_BY_SUFFIX.values())))

# 依赖缺失时的安装提示（写成可直接复制执行的命令）。
# ⚠ 每一条都必须与 `pyproject.toml` 的 optional-dependencies **实际内容**一致：
#   之前 pdf / docx 两条都写着 `rag-min[loaders]`，而该 extra 里只有 beautifulsoup4——
#   用户照着装完问题一模一样，这是最伤信任的一类错误。
#   PDF 没有"装个包就能好"的选项（见 PDF_GUIDANCE），所以这里**不给**它安装提示。
INSTALL_HINT = {
    "docx": 'pip install "rag-min[loaders]"（含 python-docx）',
    "html": 'pip install "rag-min[loaders]"（含 beautifulsoup4）',
}


@dataclass
class ConvertResult:
    """转换结果契约。

    `markdown` 是唯一被下游消费的字段；其余字段用于**观测与排障**：
    · `kind`      实际识别出的类型（便于确认分派是否正确）
    · `warnings`  过程性提示（空文本、缺依赖、跳过页等）——必须被日志记录
    · `available` 依赖是否齐备（False 时 markdown 为空，调用方应跳过该文件）
    · `reason`    为什么不可用/为什么质量下降，取值见 REASON_* 常量

    `available` 与 `reason` **必须并存**，不能用前者代替后者：
    "装了包就能好"（dep_missing）与"怎么装都没用"（not_implemented）
    对用户的行动指引完全相反，而 `available` 只能表达"能不能用"。
    """

    markdown: str = ""
    kind: str = ""
    source: str = ""
    warnings: List[str] = field(default_factory=list)
    available: bool = True
    meta: Dict[str, object] = field(default_factory=dict)
    reason: str = ""            # 空串 = 无异常（正常转换或仅是内容为空）

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()

    @property
    def is_degraded(self) -> bool:
        """是否走了降级路径：**内容可用，但结构有损**。

        单独提供一个属性而不是让调用方比较 `reason == REASON_DEGRADED`：
        调用方真正关心的是"要不要给用户一个'效果会差些'的提示"，
        把判断收在一处，以后新增降级原因时不必到处改。
        """
        return self.reason == REASON_DEGRADED


# ============================== 编码处理 ==============================

def strip_bom(text: str) -> str:
    """去掉 UTF-8 BOM。

    BOM（`\\ufeff`）会粘在文件第一个字符上，导致：
    · 第一个标题匹配不上（`#` 前多了个不可见字符）→ 面包屑首节丢失
    · 检索时把 BOM 也当成内容的一部分
    Python 的 `utf-8-sig` 解码会自动去掉它，但用 `errors="replace"` 兜底时不会，
    所以这里显式再剥一次。

    >>> strip_bom("\\ufeff# 标题")
    '# 标题'
    """
    return (text or "").lstrip("\ufeff")


def decode_text(raw: bytes) -> Tuple[str, str]:
    """把字节解码成文本，返回 (文本, 使用的编码)。

    顺序刻意如此：
        1. **UTF-8 优先** —— 现代文档的主流；纯 ASCII 也会走这条（两者兼容）
        2. **GB18030 兜底** —— 中文 Windows 上大量 `.txt` 是 GBK/GB18030 编码。
           GB18030 是 GBK 的超集，用它一条就能覆盖 GBK/GB2312，无需再探测
        3. `errors="replace"` —— 最后的兜底：宁可出现少量替换字符，也不要抛异常
           让整批索引中断

    为什么不用 chardet/charset-normalizer：额外依赖，且核心路径坚持零依赖；
    而"UTF-8 → GB18030"这两步已覆盖中文场景的绝大多数情况。

    >>> decode_text("中文".encode("utf-8"))[0]
    '中文'
    >>> decode_text("中文".encode("gb18030"))[0]
    '中文'
    """
    if raw.startswith(b"\xef\xbb\xbf"):                 # UTF-8 BOM
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # 不是合法 UTF-8 → 按中文常见编码再试一次
        try:
            return raw.decode("gb18030"), "gb18030"
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


# ============================== TXT：标题识别 ==============================

# 常见编号式标题。刻意保持**窄**：误判标题比不识别更糟——
# 错误的层级会污染面包屑，让检索结果张冠李戴。
HEADING_PATTERNS = (
    re.compile(r"^第[一二三四五六七八九十百零\d]+[章节篇部分]\s*[、:：.．]?\s*\S"),
    re.compile(r"^[（(][一二三四五六七八九十\d]+[)）]\s*\S"),
    re.compile(r"^\d+(\.\d+){1,3}\s*[、:：.．]?\s*\S"),   # 1.1 / 1.1.2
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*\S"),   # 一、 / 二.
    re.compile(r"^[IVXLC]+[.、]\s*\S"),                  # 罗马数字
    re.compile(r"^附\s*录\s*[A-Z\d一二三四五六七八九十]*\s*$"),
)

# 行尾若出现这些字符，说明这是一个完整句子 → 不该与下一行合并
SENTENCE_TAIL = "。！？；…!?;:：\"'）)》】」』"
# 列表项识别。**必须与"编号式标题"严格区分**，因为两者的正则都能命中同一批行：
#
#     `- 项` / `* 项`            → 列表（符号开头）
#     `1. 项` / `1) 项`          → 列表（数字 + 标点；`(?!\d)` 排除 `1.1` 这类多级编号）
#     `（1）项` / `(1) 项`       → 列表（**阿拉伯数字**括号编号）
#     `一、总体说明`              → **标题**（裸中文编号 + 顿号，中文技术文档的章节惯例）
#     `（三）注意事项`            → **标题**（**中文数字**括号编号）
#     `1.1 项`                   → **标题**（多级编号）
#
# 判据是"编号的写法"而不是"有没有空格"：中文数字（一二三…）在文档里几乎总是层级
# 标记，阿拉伯数字在括号里则多用于枚举。实测发现"括号后有无空格"分辨不出这两者
# （`（1）小项` 与 `（三）注意事项` 都可能没有空格），所以按数字类型区分。
# 这些约定都在 tests/test_converters.py 里钉住，改动必须是有意识的。
LIST_PREFIX = re.compile(
    r"^\s*(?:[-*+•·]"
    r"|\d+[.)、](?!\d)\s*"                   # 1. / 1) / 1、
    r"|[（(]\d+[)）]\s*"                      # （1） / (1)
    r")")


def looks_like_heading(raw_line: str, avg_len: float, line_ends_paragraph: bool) -> bool:
    """保守判断某一行是否是标题。

    `line_ends_paragraph` 表示**该行之后就是段落边界**（原文里紧跟一个空行）。
    这个参数不能省，也不能用"下一行内容为空"来替代——因为块内最后一行的
    "下一行"在数据上并不存在，把它当成"后面是空行"，会立刻退化成
    "任何短行都是标题"（实测：`hello\\nworld` 被切成 段落'hello' + 标题'world'）。

    必须同时满足（任何一条不满足即判定"不是标题"）：
        1. 行很短：<= 60 字符
        2. 不以句末标点结尾（标题通常没有句号）
        3. 不是列表项
        4. **该行之后是段落边界**
        5. 编号模式命中，或（非常短 <= 20 且 <= 平均行长 * 0.6）

    为什么这么保守：把正文误判成标题会**污染面包屑**——后续所有块的 heading
    都带上这个错误层级，检索结果溯源时会指向错误的小节。宁可漏判。
    """
    line = raw_line.strip()
    if not line or len(line) > 60:
        return False
    if line[-1] in SENTENCE_TAIL:
        return False
    if LIST_PREFIX.match(line):
        return False
    if not line_ends_paragraph:
        # 行尾没有段落边界 → 它是连续正文中的一行（硬换行），不可能是标题
        return False

    if any(p.match(line) for p in HEADING_PATTERNS):
        return True
    short_limit = max(20.0, avg_len * 0.6)
    return len(line) <= 20 and len(line) <= short_limit


# ============================== TXT：段落重排 ==============================

def reflow_paragraphs(text: str, separators: Tuple[str, ...] = ("\n",)) -> List[Tuple[str, str]]:
    """把纯文本切成 [(kind, block)]，kind ∈ {"heading", "paragraph", "list"}。

    ★ 这一段是 TXT 处理的核心，解决两个现实问题：

    **问题 1：硬换行 vs 段落换行**
        手工排版的 txt 用空行分段；机器导出的 txt（PDF 复制、日志、爬虫）常常
        每行都很短且没有空行。若只认空行，后者会挤成一个巨型段落，
        切分时只能按字符硬切，**切点落在句子中间**。
        处理方式：**先按空行切块，再把块内的单行拼接成一段**——因为块内的换行
        基本都是排版换行（reflow 的标准做法）。

    **问题 2：中英文拼接的空格**
        英文硬换行时，行尾要补空格（`hello` + `world` → `hello world`）；
        中文拼接时**不能**补空格（`中文` + `换行` → `中文换行`）。
        判断依据：看边界两侧是否都是 ASCII 可见字符。

    ⚠ 读下面的示例时请注意："拼接"只发生在**块内部**。块的最后一行若很短，
    会被 `looks_like_heading` 判定为标题（示例里的"第二行"就是这种情况）——
    这是刻意的保守策略：把正文误判成标题会污染**后续所有块**的面包屑，
    而漏判只是让块少一层结构。二者代价不对称，所以宁可漏判。

    >>> reflow_paragraphs("第一行\\n第二行\\n\\n新段落\\n")
    [('paragraph', '第一行'), ('heading', '第二行'), ('paragraph', '新段落')]
    >>> reflow_paragraphs("hello\\nworld")
    [('paragraph', 'hello world')]
    """
    blocks: List[Tuple[str, str]] = []
    if not text.strip():
        return blocks

    lines = text.splitlines()
    # 变量名用 line 而不是单字母 l：`l` 与数字 1、字母 I 在多数字体里几乎同形，
    # 是静态检查会明确警告的可读性问题（E741）
    non_blank = [line for line in lines if line.strip()]
    avg_len = sum(len(line) for line in non_blank) / max(1, len(non_blank))

    # 按空行切块，并**记录每行之后是否为段落边界**——
    # 这是标题判定的关键输入：块内除最后一行外都不是段落边界。
    # 用 `line_ends_paragraph` 显式表达，而不是让判定函数去猜"下一行是否为空"。
    chunks: List[List[Tuple[str, bool]]] = []
    current: List[Tuple[str, bool]] = []
    for idx, line in enumerate(lines):
        if line.strip():
            # ★ 只有**确实存在且为空**的下一行才算段落边界。
            #   **EOF 不算**——文件末尾只是"没有下一行了"，它对是否分段不提供信息。
            #   这一条曾经写错（把 idx+1 >= len(lines) 也算作边界），后果是
            #   硬换行文本的最后一行会被当成"独占一行的标题"：
            #   实测 `hello\nworld` 被切成 段落'hello' + 标题'world'。
            #   宁可漏判标题，也不要误判——误判会污染面包屑。
            ends_para = idx + 1 < len(lines) and not lines[idx + 1].strip()
            current.append((line, ends_para))
        else:
            if current:
                chunks.append(current)
                current = []
    if current:
        chunks.append(current)

    for chunk in chunks:
        # 块内逐行判断：标题独立成块，其余行合并成一个段落
        buffer: List[str] = []

        def flush_buffer():
            if buffer:
                joined = _join_lines(buffer)
                if joined:
                    blocks.append(("paragraph", joined))
                buffer.clear()

        for raw, ends_para in chunk:
            if looks_like_heading(raw, avg_len, ends_para):
                flush_buffer()                      # 标题前的内容先落盘
                blocks.append(("heading", raw.strip()))
            elif LIST_PREFIX.match(raw):
                flush_buffer()                      # 列表项各自独立成块
                blocks.append(("list", raw.strip()))
            else:
                buffer.append(raw.strip())
        flush_buffer()

    return blocks


def _join_lines(lines: List[str]) -> str:
    """把块内的多行拼成一段，按需补空格（英文补、中文不补）。

    判断依据是**边界两侧是否为 ASCII 可见字符**，且要看"上一个**非空格**字符"：
    行首行尾常有尾随/前导空格，若直接看 `out[-1]`，就会出现
    `abc` + ` ABC` → 先补一个空格、再把原文空格也带进来 → 双空格。
    """
    out = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not out:
            out = stripped
            continue
        need_space = ord(out[-1]) < 128 and ord(stripped[0]) < 128
        # ⚠ 分隔符拼在表达式**内部**，不要写成 f"{out} {' ' if …}{stripped}"——
        #   `{out}` 与 `{` 之间的那个字面空格会被保留，再叠加表达式产出的空格，
        #   结果就是双空格（实测：`hello  world`）。
        separator = " " if need_space else ""
        out = f"{out}{separator}{stripped}"
    return out.strip()


# ============================== 各格式转换器 ==============================

def convert_txt(path: str, source_label: str = "") -> ConvertResult:
    """TXT → Markdown。纯标准库实现，无第三方依赖。

    产出结构：
        · 识别出的标题 → `# 标题`（并按出现顺序统一为 `##`，留 `#` 给文件名）
        · 段落         → 空行分隔的普通段落
        · 列表项       → 原样保留（Markdown 与纯文本的列表语法基本兼容）

    为什么标题统一降一级（`##` 而非 `#`）：
    因为**文件名会被当作根节点面包屑**（见 `build_parent_child(source_label=)`），
    这样 `[接口文档 > 第三章 > 配置]` 才是完整路径。若标题直接用 `#`，
    就会与文件名同级，层级关系丢失。
    """
    raw = Path(path).read_bytes()
    text, encoding = decode_text(raw)
    text = strip_bom(text)
    result = ConvertResult(source=path, kind="text", meta={"encoding": encoding})

    if not text.strip():
        result.warnings.append(f"文件内容为空：{path}")
        return result

    blocks = reflow_paragraphs(text)
    if not blocks:
        result.warnings.append(f"重排后没有任何内容：{path}")
        return result

    lines: List[str] = []
    for kind, block in blocks:
        if kind == "heading":
            lines.append(f"## {block}")
            lines.append("")
        else:
            lines.append(block)
            lines.append("")

    result.markdown = "\n".join(lines).strip() + "\n"
    n_heading = sum(1 for k, _ in blocks if k == "heading")
    result.meta.update({"blocks": len(blocks), "headings": n_heading,
                        "paragraphs": sum(1 for k, _ in blocks if k == "paragraph")})
    if n_heading == 0:
        # 没有识别到标题不是错误（很多 txt 确实没有），但要留下痕迹：
        # 此时面包屑将回退为文件名（由 build_parent_child 的 source_label 提供）
        result.warnings.append(
            f"未识别到标题，面包屑将回退为文件名：{Path(path).name}")
    return result


def _normalize_markdown(text: str) -> str:
    """统一收尾：非空则以单个换行结尾。

    为什么需要：下游 `chunkers` 是按行解析 Markdown 的，
    末尾缺换行会让"最后一个块"与后续拼接内容粘在一起（表现为某段莫名变长）。
    """
    stripped = text.strip()
    return f"{stripped}\n" if stripped else ""


# ============================== HTML ==============================

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_HTML_BLOCK_TAGS = _HEADING_TAGS + ("p", "li", "pre")

# 整块删除的"页面外壳"标签。为什么要删：导航与页脚几乎出现在每个页面，
# 里面的"首页 / 关于我们 / 版权声明"会变成正文参与检索，污染大量查询的结果；
# 而 `<header>` 里通常只是站点名（同一份文档的正文标题并不在那里）。
_HTML_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript")

# 降级路径用的正则。三个模式的**顺序不能颠倒**：
#   ① 先整块删噪声 —— 必须在②之前，否则 <script> 里的 JS 会被当正文留下来
#   ② 块级标签换成换行 —— 段落边界是切分的前提（全换空格会让整篇挤成一个巨型段落）
#   ③ 剩下的行内标签换空格 —— 保留词与词之间的分隔
_NOISE_BLOCK = re.compile(
    r"<(script|style|nav|header|footer|aside|noscript)\b.*?</\1\s*>", re.S | re.I)
_BLOCK_TAG = re.compile(
    r"</?(?:p|div|br|li|tr|h[1-6]|section|article|ul|ol|dl|dt|dd|table|blockquote|pre)\b[^>]*>",
    re.I)
_ANY_TAG = re.compile(r"<[^>]+>")


def strip_html_tags(html_text: str) -> str:
    """正则去标签：**没有 beautifulsoup4 时的降级实现**（纯标准库）。

    ★ 这是"降级"而不是"不可用"：会丢标题层级与代码块围栏，但正文仍进得了索引。
      二者相比，"少层级"远好于"搜不到"——本模块历史上正是在这里把降级写成了丢弃。

    与更早的版本相比有三处改动，都是为了让降级结果**仍然可切分**：
        · 块级标签换成**换行**而不是空格。相邻的 `</p><p>` 因此形成一个空行——
          那正是 Markdown 的段落边界。若全换成空格，整篇会挤成一个巨型段落，
          切分只能按字符硬切、切点落在句子中间。
        · 用 `html.unescape` 还原实体。否则索引里存的是 `&amp;` 这类字面量，检索命中不了。
        · 同样剔除 nav/header/footer/aside，理由见 `_HTML_NOISE_TAGS`。

    >>> strip_html_tags("<p>甲</p><p>乙</p>")
    '甲\\n\\n乙'
    >>> strip_html_tags("<script>var a=1;</script><p>正文</p>")
    '正文'
    >>> strip_html_tags("<p>a &amp; b</p>")
    'a & b'
    """
    text = _NOISE_BLOCK.sub(" ", html_text)
    text = _BLOCK_TAG.sub("\n", text)
    text = _ANY_TAG.sub(" ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)          # 行内连续空白折叠成一个空格
    text = re.sub(r" *\n *", "\n", text)               # 去掉每行的首尾空格
    text = re.sub(r"\n{3,}", "\n\n", text)             # 最多保留一个空行
    return text.strip()


def _load_bs4() -> Optional[type]:
    """返回 BeautifulSoup 类；未安装时返回 None。

    为什么单独抽成一个函数，而不是在 `convert_html` 里直接 `try/except`：
    测试需要一个**确定性的**"没有 bs4"环境。写在函数体内部的话，降级分支就只能靠
    "把依赖卸载掉"来覆盖——而在装了 bs4 的开发机上，那条分支将永远测不到。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    return BeautifulSoup


def _inline_text(element) -> str:
    """提取元素内的纯文本，按"中英文边界"规则决定是否补空格。

    为什么不直接用 `element.get_text(" ")`：中文里 `<b>`、`<span>` 这类行内标签很常见，
    统一补空格会把"中文加粗继续"变成"中文 加粗 继续"——
    中文检索是按字或子词匹配的，凭空多出来的空格是纯噪声。
    规则与 TXT 段落重排的 `_join_lines` 保持一致：只在**两侧都是 ASCII** 时补空格。
    """
    out = ""
    for piece in element.strings:
        piece = " ".join(piece.split())
        if not piece:
            continue
        if not out:
            out = piece
            continue
        need_space = ord(out[-1]) < 128 and ord(piece[0]) < 128
        out = f"{out}{' ' if need_space else ''}{piece}"
    return out


def _fence(pre_element) -> str:
    """把 `<pre>` 包成 Markdown 围栏，并尽量带上语言标注。

    围栏长度自适应：正文里若本身含有 ```（例如文档正在讲 Markdown 语法），
    三个反引号会被提前闭合，后面的内容就漏成普通文本了，所以改用四个。
    """
    code = pre_element.get_text().replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    fence = "```" if "```" not in code else "````"

    classes: List[str] = []
    code_element = pre_element.find("code")
    if code_element is not None:
        classes.extend(code_element.get("class") or [])
    classes.extend(pre_element.get("class") or [])
    lang = ""
    for cls in classes:
        matched = re.match(r"(?:language|lang|highlight)-(.+)", cls)
        if matched:
            lang = matched.group(1)
            break
    return f"{fence}{lang}\n{code}\n{fence}"


def _html_to_markdown(html_text: str, soup_class) -> Tuple[str, Dict[str, int]]:
    """用 BeautifulSoup 把 HTML 转成 Markdown，返回 (markdown, 统计)。

    只处理**语义块**（标题 / 段落 / 列表项 / 代码块），行内标签交给 `_inline_text`。
    理由：切分层只认 Markdown 的块级语法，`<b>`、`<span>`、`<a>` 对它毫无意义——
    但如果把行内标签也当块处理，句子会被拆得支离破碎。

    直接接收 `soup_class` 而不是自己 import，是为了让"结构化路径"可被注入替身测试。
    """
    soup = soup_class(html_text, "html.parser")

    for noise in soup.find_all(_HTML_NOISE_TAGS):
        noise.decompose()                       # 整块删除，不留下文字

    lines: List[str] = []
    stats = {"headings": 0, "paragraphs": 0, "items": 0, "code_blocks": 0}

    for element in soup.find_all(_HTML_BLOCK_TAGS):
        # 只处理**最外层**的语义块：`<li><p>…</p></li>` 这类嵌套若两层都处理，
        # 同一段文字会被输出两次——重复内容会直接污染检索（同一段白占多个名次）
        if element.find_parent(_HTML_BLOCK_TAGS) is not None:
            continue

        if element.name == "pre":
            if element.get_text().strip():
                lines.append(_fence(element))
                stats["code_blocks"] += 1
            continue

        content = _inline_text(element)
        if not content:
            continue

        if element.name in _HEADING_TAGS:
            # 统一**降一级**：`#` 留给文件名根节点，面包屑才是
            # [文件名 > 一级标题 > 二级标题] 的完整路径（与 convert_txt 同规则）
            level = min(int(element.name[1]) + 1, 6)
            lines.append(f"{'#' * level} {content}")
            stats["headings"] += 1
        elif element.name == "li":
            lines.append(f"- {content}")
            stats["items"] += 1
        else:
            lines.append(content)
            stats["paragraphs"] += 1

    return _normalize_markdown("\n\n".join(lines)), stats


def convert_html(path: str, source_label: str = "") -> ConvertResult:
    """HTML → Markdown：装了 beautifulsoup4 走结构化，没装则降级为正则去标签。

    ★ 关键行为约定（本模块历史上正是在这里出过 bug）：
      **两条路径都返回 `available=True`**。HTML 是能读的——以前这里返回
      `available=False`，而该标志在 `pipeline.index` 里的含义是"跳过**且不登记指纹**"，
      于是 `.html` 被整篇丢弃：语料台账显示"1 篇文档"，索引却是 0 块，只留一行日志。
      **"降级"绝不等于"不可用"**，两条路径的区别只在 `reason`。

    标题降一级的原因同 `convert_txt`：文件名会被当作根节点面包屑
    （见 `build_parent_child(source_label=)`），若标题直接用 `#` 就和文件名同级了。
    """
    raw = Path(path).read_bytes()
    # 用 decode_text 而不是直接 utf-8：中文站的 HTML 相当比例是 GBK/GB18030 编码，
    # 按 UTF-8 硬读会得到一片替换字符（旧实现用 errors="replace" 会静默录入乱码）
    text, encoding = decode_text(raw)
    text = strip_bom(text)
    result = ConvertResult(source=path, kind="html", meta={"encoding": encoding})

    if not text.strip():
        result.warnings.append(f"文件内容为空：{path}")
        return result

    soup_class = _load_bs4()
    if soup_class is None:
        result.markdown = _normalize_markdown(strip_html_tags(text))
        result.reason = REASON_DEGRADED
        result.meta["degraded"] = True
        result.warnings.append(
            f"HTML 走正则降级路径（标题层级与代码块围栏丢失，面包屑将回退为文件名）："
            f"{Path(path).name}；安装 beautifulsoup4 可保留结构：{INSTALL_HINT['html']}")
        return result

    result.markdown, stats = _html_to_markdown(text, soup_class)
    result.meta.update(stats)
    if stats["headings"] == 0:
        # 没有标题不是错误（很多 HTML 片段确实没有），但要留下痕迹：
        # 此时面包屑只能回退为文件名，检索溯源会粗一档
        result.warnings.append(
            f"未识别到标题，面包屑将回退为文件名：{Path(path).name}")
    return result


def _dep_missing(kind: str, path: str, lib: str) -> ConvertResult:
    """构造"依赖缺失"的结果：不抛异常，而是返回可被日志记录、可继续批处理的提示。"""
    hint = INSTALL_HINT.get(kind, "请安装对应解析依赖")
    message = (f"{kind} 解析需要 {lib}，当前环境未安装：{Path(path).name}；"
               f"安装：{hint}")
    logger.warning("%s", message)
    return ConvertResult(source=path, kind=kind, available=False, markdown="",
                         reason=REASON_DEP_MISSING, warnings=[message])


def convert_pdf(path: str, source_label: str = "") -> ConvertResult:
    """PDF → Markdown。**刻意不实现**：明确拒绝，并给出离线转换指引。

    为什么不做（三条理由都指向同一个结论）：
        · **结构**：PDF 只有版式坐标，没有语义层级；而标题正是本项目面包屑与溯源的骨架。
          公开基准里最好的 docling 标题识别也只有 0.824（pymupdf4llm 0.412、markitdown 0.000）——
          而**错误的面包屑比没有面包屑更糟**，它会让检索结果张冠李戴。
        · **依赖**：docling 会拖进模型权重，直接推翻 `pyproject.toml` 里 `dependencies = []`
          这条"核心路径零依赖"的设计约束。
        · **覆盖面**：大量 PDF 是扫描件或加密件，根本没有文本层，需要的是 OCR（另一个课题）。

    ⚠ 这里**不再 `import docling`**：以前装了 docling 会让 warning 变成"已安装但未接入"，
      看起来像有进展，实际行为完全一样（照样跳过）。现在无论装没装任何包，
      行为与提示都是**恒定**的——"行为随环境悄悄变化"本身就是最难排查的一类坑。
    """
    message = f"PDF 不支持直接索引：{Path(path).name}。{PDF_GUIDANCE}"
    logger.warning("%s", message)
    return ConvertResult(source=path, kind="pdf", available=False, markdown="",
                         reason=REASON_NOT_IMPLEMENTED, warnings=[message])


def _doc_unsupported(path: str) -> ConvertResult:
    """.doc（Word 97-2003 二进制格式）→ 明确拒绝 + 另存指引。

    ⚠ 刻意把 `.doc` 登记进后缀表（而不是让它"扫不到"）：这样用户放进语料目录的 `.doc`
      会**被看见并被告知怎么做**。否则表现是"文件明明在目录里，却什么都没发生"——
      这是最让人困惑、也最容易误判成"程序坏了"的失败模式。
    """
    message = (f"不支持 .doc（Word 97-2003 的二进制格式）：{Path(path).name}；"
               f"该格式没有公开稳定的结构，python-docx 也读不了。"
               f"请在 Word / WPS 里「另存为」.docx 后重新索引")
    logger.warning("%s", message)
    return ConvertResult(source=path, kind="doc", available=False, markdown="",
                         reason=REASON_NOT_IMPLEMENTED, warnings=[message])


# ============================== DOCX ==============================

# 标题样式名：英文 `Heading 1` 与中文 `标题 1` 都要认（理由见 `_docx_heading_level`）
_DOCX_HEADING_RE = re.compile(r"(?:heading|标题)\s*([1-9])", re.I)
# 列表样式：英文 `List Bullet` / `List Number`，中文 Word 里是「列表段落」「项目符号」
_DOCX_LIST_RE = re.compile(r"(?:list\s*(?:bullet|number|paragraph)|列表|项目符号)", re.I)


def _load_docx() -> Optional[Callable[..., Any]]:
    """返回 python-docx 的 `Document` 工厂；未安装返回 None（理由同 `_load_bs4`）。

    ⚠ 类型标注是 `Callable` 而不是 `type`：`docx.Document` 是一个**工厂函数**
      （内部再按 路径/流 重载），并不是类。标成 `type` 会被静态检查直接拦下。
    """
    try:
        from docx import Document
    except ImportError:
        return None
    return Document


def _style_name(paragraph) -> str:
    """安全取样式名。

    为什么要 try/except：样式缺失或引用损坏的文档是真实存在的（常见于第三方工具生成的文件），
    此时 `paragraph.style` 会抛 KeyError。整篇文档不该因为一个坏样式而转换失败。
    """
    try:
        style = paragraph.style
        return (style.name or "") if style is not None else ""
    except Exception:                                  # noqa: BLE001
        return ""


def _docx_heading_level(paragraph) -> int:
    """段落是标题时返回级别（1–9），否则返回 0。

    ★ 为什么要同时看**样式名**和 **outlineLvl**：
      `style.name` 来自文档自己的 styles.xml，而**中文版 Word 保存的文档里，
      内置样式名可能是「标题 1」而不是「Heading 1」**。只匹配英文的话，中文文档的标题会被
      全部漏掉——而且是静默的：不报错，只是所有块的面包屑都退回文件名。
      `w:pPr/w:outlineLvl` 是 Word 自己记录的大纲级别，与界面语言无关，用作第二判据。
    """
    matched = _DOCX_HEADING_RE.search(_style_name(paragraph))
    if matched:
        return int(matched.group(1))
    try:
        from docx.oxml.ns import qn
        # `_p` 是私有属性，但这是官方文档给出的访问底层 XML 的方式。
        # 用 qn() 拼限定名而**不手写** "{http://...}outlineLvl"：命名空间 URI 一旦变化，
        # 手写的字符串不会报错，只会静默匹配不上。
        p_pr = paragraph._p.find(qn("w:pPr"))
        node = p_pr.find(qn("w:outlineLvl")) if p_pr is not None else None
        raw = node.get(qn("w:val")) if node is not None else None
        if raw is not None:
            return int(raw) + 1                        # outlineLvl 是 0-based，级别从 1 起
    except Exception:                                  # noqa: BLE001 取不到就当作没有级别
        pass
    return 0


def _docx_blocks(document) -> Iterator[Tuple[str, Any]]:
    """按**文档真实顺序**产出 `("paragraph" | "table", 元素)`。

    ★ 为什么不能先 `for p in doc.paragraphs` 再 `for t in doc.tables`：
      这两个是互不相干的全量列表，分别遍历会把**所有表格都挪到文末**——
      而表格往往正是"参数表 / 配置项"这类问题的答案所在；位置一乱，
      表格与它周围说明文字的上下文就断了（父子块按相邻关系聚合时会直接错配）。
      正确做法是遍历 `body` 的直接子元素，按出现顺序分派（python-docx 官方 recipe）。
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield "paragraph", Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield "table", Table(child, document)


def _docx_cell_text(cell) -> str:
    """单元格文本压成一行：Markdown 表格里不能有换行，且 `|` 必须转义。"""
    return " ".join((cell.text or "").split()).replace("|", "\\|")


def _docx_table_to_markdown(table) -> str:
    """表格 → Markdown 管道表格。

    ⚠ 已知限制：**合并单元格会被重复输出**。python-docx 对横向合并的 `row.cells`
      会多次返回同一个 cell 对象。真正还原合并需要分析 `w:gridSpan` / `w:vMerge`，
      复杂且收益有限——宁可如实重复，也不做一半的"猜"：猜错会把内容挂到错误的列上，
      而错的表格比重复的表格危险得多。
    """
    rows = [[_docx_cell_text(cell) for cell in row.cells] for row in table.rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    # 补齐参差的列数：Markdown 表格列数不一致会整张渲染错位
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def convert_docx(path: str, source_label: str = "") -> ConvertResult:
    """DOCX → Markdown。装了 python-docx 即生效（无其他依赖）。

    三个实现要点，都是"看起来能跑但结果是错的"的坑：
        · **按 paragraph 聚合，不能按 run**：同一段落里改过一次字体就会被拆成多个 run，
          按 run 输出会把一句话碎成好几段。
        · **标题样式名有中文与英文两种写法**，见 `_docx_heading_level`。
        · **表格必须按文档顺序插入**，见 `_docx_blocks`。

    标题同样**降一级**（`Heading 1` → `##`）：`#` 留给文件名根节点，面包屑才是
    [文件名 > 一级标题 > 二级标题] 的完整路径——与 txt / html 的规则保持一致。
    """
    document_class = _load_docx()
    if document_class is None:
        return _dep_missing("docx", path, "python-docx")

    try:
        document = document_class(path)
    except Exception as error:                         # noqa: BLE001
        # 损坏 / 加密 / 后缀伪装：明确报"读不了"并继续处理其他文件。
        # 不让异常冒到索引循环外面——一个坏文件不该让整批索引失败。
        message = (f"DOCX 解析失败（文件可能损坏、被加密，或后缀不是真正的 .docx）："
                   f"{Path(path).name}；{type(error).__name__}: {error}")
        logger.warning("%s", message)
        return ConvertResult(source=path, kind="docx", available=False, markdown="",
                             reason=REASON_PARSE_ERROR, warnings=[message])

    lines: List[str] = []
    stats = {"headings": 0, "paragraphs": 0, "items": 0, "tables": 0}
    for kind, block in _docx_blocks(document):
        if kind == "table":
            table_md = _docx_table_to_markdown(block)
            if table_md:
                lines.append(table_md)
                stats["tables"] += 1
            continue
        content = (block.text or "").strip()
        if not content:
            continue
        level = _docx_heading_level(block)
        if level:
            lines.append(f"{'#' * min(level + 1, 6)} {content}")
            stats["headings"] += 1
        elif _DOCX_LIST_RE.search(_style_name(block)):
            lines.append(f"- {content}")
            stats["items"] += 1
        else:
            lines.append(content)
            stats["paragraphs"] += 1

    markdown = _normalize_markdown("\n\n".join(lines))
    result = ConvertResult(source=path, kind="docx", markdown=markdown)
    # 用 update 而不是把 stats 直接传给 meta：`Dict` 在值类型上是**不协变**的，
    # `Dict[str, int]` 不能当 `Dict[str, object]` 用（mypy 会直接报 arg-type）。
    result.meta.update(stats)
    if not markdown:
        result.warnings.append(f"转换后没有任何内容：{path}")
    elif stats["headings"] == 0:
        result.warnings.append(
            f"未识别到标题，面包屑将回退为文件名：{Path(path).name}")
    return result


# ============================== 统一入口 ==============================

def detect_kind(path: str) -> str:
    """按后缀判断文档类型；未知后缀返回空串。

    >>> detect_kind("a.txt"), detect_kind("a.PDF"), detect_kind("a.bin")
    ('text', 'pdf', '')
    """
    return KIND_BY_SUFFIX.get(Path(path).suffix.lower(), "")


def is_supported(path: str) -> bool:
    return bool(detect_kind(path))


def index_capability(path: str) -> Tuple[bool, str]:
    """这个文件**在当前环境里**能否进索引，以及原因码。

    只依赖后缀与已安装的包，**不读文件内容**——所以它可以被语料台账直接调用
    （对每个文件调一次也不会带来 I/O 开销）。

    ★ 为什么需要它：语料台账必须提前告诉用户"哪些文件不会参与索引"。
      以前这个信息完全不可见——把 `.pdf` 数成"1 篇文档"、索引却是 0 块，用户只能靠猜。
      把"能不能索引"的判断收在**一处**，避免 corpus / webui / pipeline 各写一套
      （三套判断必然会在某次改动后互相不一致）。

    返回值：
        `(True,  "")`                      正常
        `(True,  REASON_DEGRADED)`         能索引但结构有损（HTML 缺 bs4）
        `(False, REASON_DEP_MISSING)`      装个包就能索引（DOCX 缺 python-docx）
        `(False, REASON_NOT_IMPLEMENTED)`  当前版本不支持（PDF / .doc）

    >>> index_capability("a.md"), index_capability("a.pdf")
    ((True, ''), (False, 'not_implemented'))
    """
    kind = detect_kind(path)
    if not kind or kind == "pdf" or kind == "doc":
        return False, REASON_NOT_IMPLEMENTED
    if kind == "docx" and _load_docx() is None:
        return False, REASON_DEP_MISSING
    if kind == "html" and _load_bs4() is None:
        return True, REASON_DEGRADED
    return True, ""


def convert(path: str, source_label: str = "") -> ConvertResult:
    """统一入口：路径 → ConvertResult（markdown 为下游唯一消费字段）。

    ⚠ 与旧行为的差异（这是一处**一致性修复**）：此前 `iter_documents` 对目录扫描
    会按后缀过滤，但传**单个文件**时不校验后缀——于是 `.pdf` 会被当纯文本读成
    乱码并静默建进索引。现在统一在这里分派，未知类型明确报错。

    Markdown 文件直接原样返回（它本身就是我们想要的中间表示）。
    """
    kind = detect_kind(path)
    label = source_label or Path(path).name

    if kind == "markdown":
        text = strip_bom(Path(path).read_text(encoding="utf-8", errors="replace"))
        return ConvertResult(markdown=text, kind=kind, source=path,
                             warnings=[] if text.strip() else [f"文件内容为空：{path}"])
    if kind == "text":
        return convert_txt(path, label)
    if kind == "pdf":
        return convert_pdf(path, label)
    if kind == "docx":
        return convert_docx(path, label)
    if kind == "doc":
        return _doc_unsupported(path)
    if kind == "html":
        # 这里过去直接返回 available=False，导致 .html 被整篇跳过（详见 convert_html）。
        # 现在统一走 convert_html：有 bs4 就结构化，没有就降级，**两条路都能进索引**。
        return convert_html(path, label)
    raise ValueError(f"不支持的文档类型：{path}（支持 {SUPPORTED_KINDS}）")
