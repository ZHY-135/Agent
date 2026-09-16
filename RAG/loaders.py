# -*- coding: utf-8 -*-
"""① 文档加载层：把各种来源统一成 (path, text) 流。

当前代码缺的一环：split_into_chunks(doc_file) 只吃单个 .md 文件。
生产环境需要：多格式（md/txt/html/pdf/docx）、目录递归、增量（跳过未变动文件）。
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from .logging_setup import get_logger

logger = get_logger(__name__)

# 必须与 converters.KIND_BY_SUFFIX **逐项保持一致**（两处不一致会导致
# "扫描得到但转换不了"或"能转换但扫描不到"这类半边可用的问题）。
# 注意其中的 `.pdf` / `.doc` 是"**登记但不支持**"：登记的目的是让它们被看见并给出
# 可执行的提示，而不是被静默忽略（见 converters.convert_pdf / _doc_unsupported）。
SUPPORTED = {".md", ".markdown", ".txt", ".html", ".htm",
             ".pdf", ".docx", ".doc"}


def iter_documents(path: str) -> Iterator[str]:
    """目录递归 / 单文件，统一产出文件路径。

    ⚠ **单文件也必须校验后缀**（本函数修过的一个静默错误）：
      目录扫描一直按 `SUPPORTED` 过滤，但单个文件路径原先直接 yield，
      于是 `python -m RAG.main index 某文档.pdf` 会把二进制 PDF
      按 UTF-8 读成乱码、当成一篇正常文档建进索引——不报错、不警告，
      只是检索结果全变成乱码。宁可**当场报错**，也不要静默录入垃圾。
    """
    if os.path.isfile(path):
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in SUPPORTED:
            raise ValueError(
                f"不支持的文件类型 {suffix or '(无后缀)'}：{path}\n"
                f"    支持的后缀：{', '.join(sorted(SUPPORTED))}\n"
                f"    需要新增格式时，请在 converters.py 里实现对应转换器，"
                f"并把后缀加入 loaders.SUPPORTED 与 converters.KIND_BY_SUFFIX"
                f"（两处必须一致）。")
        yield os.path.abspath(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in SUPPORTED:
                    yield os.path.abspath(os.path.join(root, fn))
    else:
        raise FileNotFoundError(f"路径不存在: {path}")


def file_fingerprint(path: str) -> str:
    """内容指纹：用于增量索引——文件没变就跳过，不重复付费 embedding。"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_text(path: str) -> str:
    """读取文本。⚠ 必须显式 encoding，否则 Windows 中文机（GBK）读 UTF-8 会崩。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if path.lower().endswith((".html", ".htm")):
        text = _strip_html(text)
    if not text.strip():
        logger.warning("文件内容为空：%s", path)
    return text


def _strip_html(html: str) -> str:
    """保留的兼容入口：委托给 `converters.strip_html_tags`（**单一真源**）。

    为什么改成委托而不是各留一份实现：正则去标签一旦有两份就会各自演化——
    本模块与 converters 曾经各有一版，行为已经不一致（一版保留段落边界、一版不保留，
    而段落边界直接决定切分质量）。实现只保留在 `converters` 里。

    依赖方向必须是 loaders → converters（`converters` 不反向依赖 `loaders`），
    否则会形成循环导入——所以这里用函数内延迟导入，与 `load_converted` 的写法一致。
    """
    from .converters import strip_html_tags
    return strip_html_tags(html)


def load_all(path: str, seen: Optional[Dict[str, str]] = None) -> Iterator[Tuple[str, str, str]]:
    """产出 (path, text, fingerprint)；传入 seen 可跳过未变动文件。

    ⚠ 这里的 `text` 是**原始读取结果**（HTML 仅做正则去标签），
    不保证是结构化 Markdown。需要"能切出面包屑"的文本请用 `load_converted()`。
    保留本函数是为了不改变既有调用契约（测试与外部使用者依赖它）。
    """
    for p in iter_documents(path):
        fp = file_fingerprint(p)
        if seen is not None and seen.get(p) == fp:
            continue                      # 增量：内容未变，跳过
        yield p, load_text(p), fp


class LoadedDoc:
    """一份已加载并转换完毕的文档。

    为什么用一个小对象而不是元组：调用方需要区分三种情况
        · 正常内容            → markdown 非空
        · 转换后为空          → markdown 为空，但**必须登记指纹**（否则每次索引都重读）
        · 依赖缺失/不可用      → available=False，应跳过且**不登记指纹**
    元组表达不了这个区别，最后会退化成"在调用方猜测"。

    `reason` 与 `available` 并存的原因见 `converters.ConvertResult`：
    "装个包就能好"与"怎么装都没用"对用户的行动指引相反，而 `available` 只能表达
    "能不能用"。界面要靠 `reason` 分流提示，所以必须一路透传到调用方。

    ⚠ 加了字段就必须**同步加进 `__slots__`**——漏掉会直接抛 AttributeError，
    而且是在赋值那一刻才炸（不是定义处），排查起来容易绕远路。
    """

    __slots__ = ("path", "markdown", "fingerprint", "label", "warnings",
                 "available", "reason")

    def __init__(self, path: str, markdown: str, fingerprint: str, label: str,
                 warnings: Optional[list] = None, available: bool = True,
                 reason: str = ""):
        self.path = path
        self.markdown = markdown
        self.fingerprint = fingerprint
        self.label = label
        self.warnings = list(warnings or [])
        self.available = available
        self.reason = reason

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()


def load_converted(path: str) -> Iterator["LoadedDoc"]:
    """产出 `LoadedDoc`——**推荐下游用这个**。

    与 `load_all` 的差别：在切分之前先经过 `converters.convert()` 做格式归一化，

        .txt / .pdf / .docx → convert() → Markdown → 交给 chunkers

    这样切分层就能拿到它唯一认识的 Markdown 语法（`#` 标题、``` 围栏、空行分段），
    从而产出标题面包屑、保护代码块。

    `label` 是**面包屑兜底用的根节点名**（取文件名主干）：
    纯文本与去标签后的 HTML 自身没有标题，若不给兜底，这些块的 heading 会是空的，
    检索结果就无法溯源到任何层级。

    ⚠ **本函数不跳过任何文件**（依赖缺失除外），也不打印警告——
    是否跳过由调用方决定（`pipeline.index()` 需要为空文件登记指纹），
    警告也由调用方记录，避免同一句话被打印两次。
    """
    from .converters import convert  # 延迟导入，避免循环依赖

    for p in iter_documents(path):
        result = convert(p)
        yield LoadedDoc(
            path=p,
            markdown=result.markdown,
            fingerprint=file_fingerprint(p),
            label=Path(p).stem,
            warnings=result.warnings,
            available=result.available,
            reason=result.reason,
        )


def load_all_converted(path: str, seen: Optional[Dict[str, str]] = None
                       ) -> Iterator["LoadedDoc"]:
    """`load_converted` + 增量跳过：指纹未变且依赖可用时直接跳过该文件。

    把增量判断放在加载层是刻意的：跳过意味着**不转换、不切分、不 embedding**，
    越早跳过省得越多（PDF 转换可能是整个索引里最贵的一步）。

    ⚠ **只比较文件内容哈希，不含"索引签名"**（切分参数 + 嵌入模型）。
    也就是说：改了 `MAX_CHILD_CHARS` 或换了 embedding 模型时，本函数**仍会跳过**
    ——那种情况下请用 `pipeline.RAGPipeline.index()` 的增量判断，它比较的是
    「内容哈希 + 索引签名」的组合指纹，参数变了会重建。
    保留本函数是为了不改变既有调用契约，新代码优先用 pipeline。
    """
    for doc in load_converted(path):
        if not doc.available:
            continue                                      # 无法转换：不登记指纹，下次再试
        if seen is not None and seen.get(doc.path) == doc.fingerprint:
            continue                                      # 内容未变：整篇跳过
        yield doc
