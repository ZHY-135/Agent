# -*- coding: utf-8 -*-
"""日志基础设施：统一格式、可开关的控制台输出。

--------------------------------------------------------------------------------
设计约定（这条约定不写清就会变成"到处都是 print"）
--------------------------------------------------------------------------------
1. **库代码只记录，不配置**：各模块用 `get_logger(__name__)` 取 logger，
   不自己 add handler、不设 level。由**应用层**（`main.py`）决定输出到哪、什么级别。
   否则任何 `import RAG` 都会污染宿主的日志配置。

2. **CLI 的用户可见输出用 print，诊断信息用 logging**。
   两者职责不同：
     · print  —— 命令的结果（答案、指标表、报告），用户就是来看这个的；
     · logging —— 过程与异常（扫了多少文件、哪条查询异常、为什么拒答），
                  默认**静默**，加 --verbose 才显示。
   混用会让 CLI 输出被日志淹没，也会让日志里混进需要被解析的结果。

3. **默认不输出任何日志**：未调用 `configure_logging()` 时走
   `logging.lastResort`（WARNING 及以上才出到 stderr）。这样作为库被引入时
   既不会静默吞掉错误，也不会刷屏。
"""
import logging
import sys

__all__ = ["get_logger", "configure_logging", "DEFAULT_FORMAT"]

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# 记录是否已配置，避免重复 add handler（重复配置会让每条日志打印多次）
_configured = False


def get_logger(name: str) -> logging.Logger:
    """取模块级 logger。库代码统一用这个，不要自己调 logging.getLogger 后加 handler。"""
    return logging.getLogger(name)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """配置根 logger。由应用层（CLI 入口）调用，库代码不应调用。

    Args:
        verbose: 输出 DEBUG 及以上（含每个文件的索引过程、每条查询的召回明细）。
        quiet:   只输出 WARNING 及以上（用于把 CLI 的输出喂给其他程序时）。
    """
    global _configured
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # 重复调用只调整级别，不重复加 handler
    if _configured:
        return

    stream = sys.stderr
    # Windows 控制台默认 GBK，中文日志按 UTF-8 写会抛 UnicodeEncodeError。
    # 与 main.py 对 stdout 的处理保持一致：能重配就重配，且 errors="replace" 兜底。
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):        # 已被重定向到不支持重配的对象
            pass

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    _configured = True
