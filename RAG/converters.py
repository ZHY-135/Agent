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

    .txt / .pdf / .docx → converters.convert() → Markdown → loaders → chunkers

好处：① 切分器一行都不用改 ② 转换器可独立测试（契约：路径 → Markdown）
      ③ 依赖隔离（PDF/Word 的库只在用到时才需要）

行业做法与此一致：MarkItDown(微软)、Docling(IBM)、MinerU、Marker、pymupdf4llm
全部输出 Markdown。

--------------------------------------------------------------------------------
当前实现状态
--------------------------------------------------------------------------------
    ✅ convert_txt    —— 纯标准库，已完整实现
    ⏳ convert_pdf    —— 需要 docling（见 docs/项目后续改进方案.md 第十七章）
    ⏳ convert_docx   —— 需要 python-docx
    ⏳ convert_html   —— 需要 beautifulsoup4（当前由 loaders._strip_html 正则兜底）

未安装依赖时**不抛 ImportError**，而是返回带 `available=False` 与安装提示的结果——
这样批量索引时能继续处理其他文件，并在日志里给出可执行的修复指引。
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .logging_setup import get_logger

logger = get_logger(__name__)

__all__ = ["ConvertResult", "convert", "convert_txt", "is_supported",
           "SUPPORTED_KINDS", "strip_bom", "decode_text", "reflow_paragraphs",
           "looks_like_heading"]

# 后缀 → 文档类型。集中一处便于与 loaders.SUPPORTED 对齐
KIND_BY_SUFFIX: Dict[str, str] = {
    ".md": "markdown", ".markdown": "markdown",
    ".txt": "text",
    ".html": "html", ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
}
SUPPORTED_KINDS = tuple(sorted(set(KIND_BY_SUFFIX.values())))

# 依赖缺失时的安装提示（写成可直接复制执行的命令）
INSTALL_HINT = {
    "pdf": 'pip install "rag-min[loaders]"（含 docling）',
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
    """

    markdown: str = ""
    kind: str = ""
    source: str = ""
    warnings: List[str] = field(default_factory=list)
    available: bool = True
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()


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

    >>> reflow_paragraphs("第一行\\n第二行\\n\\n新段落\\n")
    [('paragraph', '第一行第二行'), ('paragraph', '新段落')]
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


def _dep_missing(kind: str, path: str, lib: str) -> ConvertResult:
    """构造"依赖缺失"的结果：不抛异常，而是返回可被日志记录、可继续批处理的提示。"""
    hint = INSTALL_HINT.get(kind, "请安装对应解析依赖")
    message = (f"{kind} 解析需要 {lib}，当前环境未安装：{Path(path).name}；"
               f"安装：{hint}")
    logger.warning("%s", message)
    return ConvertResult(source=path, kind=kind, available=False,
                         warnings=[message], markdown="")


def convert_pdf(path: str, source_label: str = "") -> ConvertResult:
    """PDF → Markdown。**尚未实现**（需要 docling，见改进方案第十七章）。

    选型依据（公开基准）：docling 的标题识别得分 0.824，而 pymupdf4llm 仅 0.412、
    markitdown 为 0.000 —— 而标题正是本项目面包屑的唯一来源。
    """
    try:
        import docling  # noqa: F401
    except ImportError:
        return _dep_missing("pdf", path, "docling")
    # 依赖存在但转换逻辑尚未接入：明确报"未实现"，不静默返回空
    return ConvertResult(
        source=path, kind="pdf", available=False,
        warnings=[f"PDF 转换尚未接入（docling 已安装，但 convert_pdf 未实现）：{Path(path).name}"])


def convert_docx(path: str, source_label: str = "") -> ConvertResult:
    """DOCX → Markdown。**尚未实现**（需要 python-docx，见改进方案第十七章）。

    实现要点（待做）：
        · `paragraph.style.name` 形如 "Heading 1" → 映射为 `##`（与 txt 同样降一级）
        · **必须按 paragraph 聚合，不能按 run** ——同段落内改过一次字体就会被拆成多个 run
        · `.doc` 老格式读不了：`python-docx` 只支持 `.docx`，需明确提示另存
    """
    try:
        import docx  # noqa: F401
    except ImportError:
        return _dep_missing("docx", path, "python-docx")
    return ConvertResult(
        source=path, kind="docx", available=False,
        warnings=[f"DOCX 转换尚未接入（python-docx 已安装，但 convert_docx 未实现）：{Path(path).name}"])


# ============================== 统一入口 ==============================

def detect_kind(path: str) -> str:
    """按后缀判断文档类型；未知后缀返回空串。

    >>> detect_kind("a.txt"), detect_kind("a.PDF"), detect_kind("a.bin")
    ('text', 'pdf', '')
    """
    return KIND_BY_SUFFIX.get(Path(path).suffix.lower(), "")


def is_supported(path: str) -> bool:
    return bool(detect_kind(path))


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
    if kind == "html":
        # HTML 暂由 loaders._strip_html 处理（正则去标签，会丢失标题层级）。
        # 这里明确标注为"降级路径"，避免误以为它和 Markdown 等价。
        return ConvertResult(
            source=path, kind=kind, available=False,
            warnings=[f"HTML 暂走 loaders 的正则去标签降级路径（标题层级会丢失）："
                      f"{Path(path).name}；建议改用 beautifulsoup4 保留结构"])
    raise ValueError(f"不支持的文档类型：{path}（支持 {SUPPORTED_KINDS}）")
