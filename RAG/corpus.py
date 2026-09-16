# -*- coding: utf-8 -*-
"""语料目录管理：路径校验与清洗、系统对话框选择、收藏、界面状态的持久化。

--------------------------------------------------------------------------------
为什么单独一个模块（而不是写在 webui.py 里）
--------------------------------------------------------------------------------
`webui.py` 顶部就 `import streamlit`，一旦逻辑写在里面，任何测试都必须先装
streamlit（一个几十 MB 的可选依赖）——于是"可选依赖"就变成了"测试的前提"。
把**纯逻辑**（校验路径、弹系统对话框、读写状态文件、收藏夹增删）放在这里，
渲染留在 `webui.py`，两边各管一件事：

    corpus.py   可离线测试、零依赖（只用标准库 + 本项目的 loaders）
    webui.py    只负责把这里的结论画出来

--------------------------------------------------------------------------------
状态文件放在哪、为什么不入库
--------------------------------------------------------------------------------
默认写到**项目根**的 `rag.local.json`（相对本文件定位，因此与启动目录无关）。
它记录"上次用的语料目录 / 收藏的目录 / 面板上那几个开关"——
属于本机个人偏好，不是项目资产，因此 `.gitignore` 已排除，且**读写失败都不抛异常**
（只读挂载、权限不足时面板应当照常可用，只是记不住设置）。
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .loaders import SUPPORTED, iter_documents

# 状态文件放项目根：与启动目录无关（用 __file__ 定位），便于"这个项目的偏好归这个项目管"
STATE_FILENAME = "rag.local.json"
MAX_FAVORITES = 20


def default_state_path() -> Path:
    return Path(__file__).resolve().parent.parent / STATE_FILENAME


def supported_suffixes_text() -> str:
    """给用户看的受支持后缀清单（从 loaders.SUPPORTED 生成，避免两处维护）。"""
    return "、".join(sorted(SUPPORTED))


def _rstrip_separators(text: str) -> str:
    """去掉末尾分隔符，但**绝不削掉"根"**。

    ★ 这里修的是一个真实缺陷：`D:\\` 曾被削成 `D:`。而在 Windows 上
      `D:` 是"**D 盘的当前目录**"（一个相对路径），不是"D 盘根"——
      更糟的是 `Path("D:").is_dir()` 会返回 True（如果 D 盘有当前目录），
      于是浏览面板点"盘符按钮"后看起来成功了，实际跳到了别的地方。
      所以判断标准不是"末尾是不是分隔符"，而是"削掉之后还剩不剩路径"：
          `D:\\`            → drive=`D:`，rest=`\\`   → rest 是根 → 不削
          `\\\\srv\\share\\`  → drive=`\\\\srv\\share`，rest=`\\` → 不削
          `D:\\docs\\`       → drive=`D:`，rest=`\\docs\\` → 削
    """
    while len(text) > 1 and text[-1] in "/\\":
        _drive, rest = os.path.splitdrive(text)
        if rest in ("\\", "/"):
            break                                     # 已经是根（盘符根 / UNC 根）
        text = text[:-1]
    return text


def normalize_dir(user_input: str) -> str:
    """把用户在输入框里敲的路径收拾干净。

    处理四种真实输入习惯（都不是"用户错了"，而是界面该容错）：
        · 从资源管理器/文档里复制路径会带双引号：`"D:\\我的文档"` → 去掉引号
        · 引号与末尾分隔符可能**同时出现且顺序不定**：`"D:\\我的文档\\"` → 两者都要去掉
          （所以下面用循环反复剥，而不是各剥一次）
        · 手写 `~/docs` 或 `%USERPROFILE%\\docs` → 展开环境变量与家目录
        · 末尾带分隔符 `D:\\我的文档\\` → 去掉，避免拼出 `\\\\` 或比较时不一致；
          但**盘符根 `D:\\` 必须保持原样**（见 `_rstrip_separators`）

    注意：**不**在这里转成绝对路径——相对路径要按"执行时的项目根"解析，
    提前转换会把用户填的 `docs` 变成一长串绝对路径，反而看不懂。
    """
    text = (user_input or "").strip()
    # 引号与分隔符会交替暴露（先把引号剥掉，末尾分隔符才露出来），
    # 循环几次直到稳定；上限 4 次足够，也避免任何意外的死循环。
    for _ in range(4):
        before = text
        text = text.strip().strip('"').strip("'").strip()
        text = _rstrip_separators(text)
        if text == before:
            break
    if not text:
        return ""
    return os.path.expandvars(os.path.expanduser(text)) or ""


@dataclass
class CorpusStatus:
    """一次语料目录校验的结论（供界面直接渲染，不需要再判断）。"""

    path: str = ""
    level: str = "error"                  # ok / warn / error
    message: str = ""
    n_files: int = 0
    total_bytes: int = 0
    files: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """能否拿去建索引。

        ⚠ warn 也算"可用"：目录里暂时没有受支持文档时，
          索引结果为空是**正确行为**（而不是错误），界面只需提醒。
        """
        return self.level in ("ok", "warn")

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "error": "❌"}.get(self.level, "❌")


def check_corpus_dir(user_input: str, sample_limit: int = 5) -> CorpusStatus:
    """校验一个语料目录，返回可直接显示的状态。

    ★ 为什么要在界面层先校验、而不是直接让 `pipeline.index()` 去撞：
      路径写错时原来的表现是抛 `FileNotFoundError`，整页变成红色异常栈；
      用户看到的是"程序坏了"，而真实原因只是**打错了一个字**。
      先把三种常见情况翻译成人话（不存在 / 传成文件 / 没有可用文档），
      再决定要不要真的去建索引。
    """
    path = normalize_dir(user_input)
    if not path:
        return CorpusStatus(level="error", message="请填写语料目录路径")

    target = Path(path)
    if not target.exists():
        return CorpusStatus(path=path, level="error",
                            message=f"路径不存在：{path}（相对路径按项目根解析）")
    if target.is_file():
        suffix = target.suffix.lower()
        hint = ("  该文件的后缀不在支持列表里，无法作为语料导入。"
                if suffix not in SUPPORTED else
                "  语料目录需要传**目录**；若只想索引这一个文件，请用 `index <文件>`。")
        return CorpusStatus(path=path, level="error",
                            message=f"这是一个文件，不是目录：{target.name}\n{hint}")

    try:
        files = list(iter_documents(path))
    except (ValueError, OSError) as error:                # 目录层面极少见，兜底不崩
        return CorpusStatus(path=path, level="error", message=f"扫描目录失败：{error}")

    total = 0
    for item in files:
        try:
            total += Path(item).stat().st_size
        except OSError:
            continue                                      # 单个文件读不到大小不影响整体判断

    if not files:
        return CorpusStatus(path=path, level="warn", n_files=0, total_bytes=0,
                            message=f"目录里没有受支持的文档（支持：{supported_suffixes_text()}）"
                                    f"；索引结果会是空——这可能是你想要的，也可能是路径填错了层级")
    return CorpusStatus(path=path, level="ok", n_files=len(files), total_bytes=total,
                        message=f"{len(files)} 个文件 / {total / 1024:.1f} KB",
                        files=[str(Path(f).name) for f in files[:sample_limit]])


# ============================== 原生"选择文件夹"对话框 ==============================

@dataclass
class PickResult:
    """一次系统对话框选择的结果（成功与否都要给用户一句话）。"""

    path: Optional[str] = None
    message: str = ""
    ok: bool = False


# 子进程里执行的脚本。它自己拥有主线程与 Tk 事件循环，选完把路径打到 stdout。
# 退出码即错误类型：3 = 没有 tkinter，4 = 没有图形环境（服务器/容器）。
_PICK_SCRIPT = r'''
import sys
try:
    import tkinter
    from tkinter import filedialog
except Exception as error:                  # 精简版 Python 可能没带 tkinter
    sys.stderr.write("NO_TK:%s" % error)
    raise SystemExit(3)

initial = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    root = tkinter.Tk()
except Exception as error:                  # 无图形环境（纯命令行服务器）
    sys.stderr.write("NO_GUI:%s" % error)
    raise SystemExit(4)

root.withdraw()                             # 只要对话框，不要多一个空白主窗口
try:
    root.attributes("-topmost", True)       # 否则容易被浏览器窗口盖住，用户会以为没反应
except Exception:
    pass
try:
    path = filedialog.askdirectory(
        title="选择语料目录（该目录会被递归索引）",
        initialdir=initial or None,
        mustexist=True)
finally:
    root.destroy()
print(path or "")
'''


def native_pick_directory(initial: str = "", timeout: float = 180.0,
                          runner: Optional[Callable[..., Any]] = None,
                          python: Optional[str] = None) -> PickResult:
    """弹出系统原生的「选择文件夹」对话框，返回用户选中的目录。

    ★ 为什么用**子进程**跑 tkinter，而不是在当前进程里直接调：
      Streamlit 的脚本运行在它自己的工作线程里，而 tkinter 要求
      "创建 Tk 的线程就是主线程并由它跑事件循环"。在工作线程里 `Tk()`，
      轻则报 `main thread is not in main loop`，重则把整个面板卡死。
      放进独立子进程后，弹窗/等待/关闭全在那个进程的主线程里完成，
      我们只读一行 stdout —— 进程隔离一次性消除这类 GUI 线程问题。

    ⚠ 对话框弹出在**运行 `serve` 的那台机器**的桌面上。若你是在别的电脑上
      通过浏览器访问面板，对话框不会出现在你面前（见 webui 里的提示）。

    `runner` / `python` 是给测试用的注入点：真弹窗无法在无人值守环境里自动化，
    但"各种返回情况如何翻译成人话"可以完整测到（见 tests/test_corpus.py）。
    """
    executable = python or sys.executable
    run = runner or subprocess.run
    try:
        completed = run([executable, "-c", _PICK_SCRIPT, normalize_dir(initial)],
                        capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return PickResult(None, f"等待超时（{timeout:.0f} 秒）：对话框可能被别的窗口挡住了，"
                                f"或者不在这台机器的桌面上", False)
    except Exception as error:                          # 解释器不存在 / 权限等
        return PickResult(None, f"无法启动选择对话框：{error}", False)

    stderr = completed.stderr or ""
    if completed.returncode == 3 or "NO_TK:" in stderr:
        return PickResult(None, "这个 Python 没带 tkinter，无法弹出系统对话框——"
                                "请在面板的目录输入框里直接粘贴路径", False)
    if completed.returncode == 4 or "NO_GUI:" in stderr:
        return PickResult(None, "当前环境没有图形界面，系统对话框只能在运行面板的"
                                "那台机器桌面上弹出——请在目录输入框里直接粘贴路径", False)
    if completed.returncode != 0:
        return PickResult(None, f"选择对话框异常退出（code={completed.returncode}）："
                                f"{stderr.strip()[:200]}", False)

    chosen = normalize_dir(completed.stdout or "")
    if not chosen:
        return PickResult(None, "已取消选择", False)
    return PickResult(chosen, f"已选择：{chosen}", True)


# ============================== 界面状态持久化 ==============================

DEFAULT_STATE: Dict[str, Any] = {
    "docs_dir": "",                       # 空 = 用 DEFAULT_DOCS / 命令行传入值
    "favorites": [],
    "use_bm25": True,
    "use_mmr": True,
    "with_generator": True,
    "top_k": 5,
    "min_sim": 0.0,
}


def load_ui_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取界面状态；任何异常都退回默认值（宁可记不住，也不能打不开面板）。

    为什么要 merge 而不是直接用读到的字典：旧版本的状态文件可能缺字段，
    升级后多了开关就会 KeyError——merge 让状态文件向后兼容。
    """
    state = dict(DEFAULT_STATE)
    target = Path(path) if path is not None else default_state_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return state
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # 文件被手动改坏 / 编码不对：直接用默认值，并且**不覆盖**它（保留现场便于排查）
        return state
    if not isinstance(raw, dict):
        return state
    for key, fallback in DEFAULT_STATE.items():
        if key in raw:
            state[key] = raw[key]
    if not isinstance(state.get("favorites"), list):
        state["favorites"] = []
    return state


def save_ui_state(state: Dict[str, Any], path: Optional[Path] = None) -> bool:
    """保存界面状态；失败返回 False（**不抛异常**，只读环境下面板仍要能用）。"""
    target = Path(path) if path is not None else default_state_path()
    payload = {key: state.get(key, fallback) for key, fallback in DEFAULT_STATE.items()}
    try:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return True
    except OSError:
        return False


def _same_path(a: str, b: str) -> bool:
    """路径比较：Windows 不区分大小写，且相对路径不能靠字符串直接比。"""
    left, right = normalize_dir(a), normalize_dir(b)
    if os.name == "nt":
        return os.path.normcase(os.path.abspath(left)) == \
            os.path.normcase(os.path.abspath(right))
    return os.path.abspath(left) == os.path.abspath(right)


def add_favorite(state: Dict[str, Any], path: str) -> Dict[str, Any]:
    """收藏一个目录（去重、保序、限长），返回**新的**状态字典。

    为什么返回新字典而不是就地改：`st.session_state` 里的嵌套容器就地修改时，
    Streamlit 有时不会把变化登记下来；每次都产生新对象最不容易踩这个坑。
    """
    cleaned = normalize_dir(path)
    favorites = [f for f in state.get("favorites") or [] if isinstance(f, str)]
    if not cleaned:
        return dict(state, favorites=favorites)
    favorites = [f for f in favorites if not _same_path(f, cleaned)]
    favorites.insert(0, cleaned)                          # 最近收藏的排最前
    return dict(state, favorites=favorites[:MAX_FAVORITES])


def remove_favorite(state: Dict[str, Any], path: str) -> Dict[str, Any]:
    favorites = [f for f in state.get("favorites") or []
                 if isinstance(f, str) and not _same_path(f, path)]
    return dict(state, favorites=favorites)


def favorite_rows(favorites: Sequence[str]) -> List[Dict[str, Any]]:
    """收藏列表 + 每个目录的实时校验结果（选择前就能看出哪个还有效）。"""
    rows = []
    for item in favorites:
        status = check_corpus_dir(item)
        rows.append({"目录": item, "状态": f"{status.icon} {status.message.splitlines()[0]}",
                     "文件数": status.n_files, "可用": status.usable})
    return rows


def state_to_json(state: Dict[str, Any]) -> str:
    """把状态渲染成可读 JSON（面板里"查看/导出设置"用）。"""
    return json.dumps({key: state.get(key, fallback) for key, fallback in DEFAULT_STATE.items()},
                      ensure_ascii=False, indent=2)


__all__ = [
    "CorpusStatus", "DEFAULT_STATE", "MAX_FAVORITES", "PickResult", "STATE_FILENAME",
    "add_favorite", "check_corpus_dir", "default_state_path",
    "favorite_rows", "load_ui_state",
    "native_pick_directory", "normalize_dir",
    "remove_favorite", "save_ui_state", "state_to_json", "supported_suffixes_text",
]
