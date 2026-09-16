# -*- coding: utf-8 -*-
"""RAG 命令行入口。

离线演示（零依赖、零费用）：
    python -m RAG.main demo               索引示例文档，走检索链路，打印 prompt
    python -m RAG.main qa "问题"          走完整问答链路（EchoGenerator，含引用校验与拒答）

持久化索引与真实生成：
    python -m RAG.main init-db
    python -m RAG.main index ./docs
    python -m RAG.main query "问题" --provider openai
"""
import argparse
import sys
import threading
from pathlib import Path
from typing import Optional

from .embedders import CachingEmbedder, HashEmbedder, OpenAIEmbedder
from .evaluation import (
    ABLATIONS,
    check_anchors,
    load_golden_set,
    render_check_report,
    render_report,
    run_experiment,
)
from .generators import EchoGenerator, Generator, OpenAIGenerator
from .loaders import iter_documents
from .logging_setup import configure_logging, get_logger
from .pipeline import RAGPipeline
from .settings import Settings
from .stores import MemoryStore, PgVectorStore

logger = get_logger(__name__)

DEMO_DOC = """# 第三章 数据库连接池

最大连接数设为200，原因是压测发现超过200后响应时间陡增。

## 配置示例

```python
def connect():
    return Pool(max=200)
```

## 常见错误

遇到 error code 0x80070005 时，通常是权限配置问题而非连接数不足。
"""


def build_embedder(settings: Settings):
    if settings.embedding_provider == "hash":
        return CachingEmbedder(HashEmbedder(settings.embedding_dim))
    if not settings.openai_api_key:
        raise ValueError("使用 OpenAI embedding 时必须设置 OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("请安装 openai：pip install openai") from error
    return CachingEmbedder(OpenAIEmbedder(
        OpenAI(api_key=settings.openai_api_key), settings.embedding_model, settings.embedding_dim
    ))


def build_store(settings: Settings):
    if settings.backend == "memory":
        return MemoryStore()
    if not settings.database_url:
        raise ValueError("使用 pgvector 时必须设置 DATABASE_URL")
    try:
        from psycopg2.pool import SimpleConnectionPool
    except ImportError as error:
        raise RuntimeError("请安装 psycopg2：pip install psycopg2-binary pgvector") from error
    return PgVectorStore(SimpleConnectionPool(1, 5, settings.database_url), settings.embedding_dim)


def build_generator(provider: str, settings: Settings, model: Optional[str] = None) -> Generator:
    """按 provider 装配生成器。

    echo   —— 零依赖：把检索到的资料按 [n] 复述，用于离线跑通生成链路与验证引用校验；
    openai —— 生产：真实 LLM 生成，同样复用拒答与引用回填校验。
    """
    if provider == "echo":
        return EchoGenerator()
    if not settings.openai_api_key:
        raise ValueError("使用 openai 生成器时必须设置 OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("请安装 openai：pip install openai") from error
    return OpenAIGenerator(OpenAI(api_key=settings.openai_api_key),
                           model=model or "gpt-4o-mini")


def build_pipeline(settings: Settings, generator: Optional[Generator] = None) -> RAGPipeline:
    return RAGPipeline(build_embedder(settings), build_store(settings),
                       use_bm25=True, use_mmr=True, generator=generator)


def ensure_demo_document(docs_dir: Path) -> None:
    """确保"演示语料"存在——**只在目录里确实没有可用文档时**才写入。

    ★ 为什么必须判断目录是否为空（实测踩到的问题）：
      用户用 `qa --docs 我的语料` 指向自己的目录时，旧实现会**无条件**往里塞一个
      `demo_rag_doc.md`。实测后果有两个，都会让人莫名其妙：
        · 用户的语料文件夹里凭空多出一个陌生文件（305 字节）；
        · 这个演示文档还会被一起索引（实测父块 6 → 9），
          既污染检索结果，也让评测的语料构成与预期不符。

      现在的规则：目录里只要已有**任一受支持格式**的文档，就什么都不做。
      这也让 `demo`（把目录当位置参数传）与 `qa --docs` 的行为都变得可预期。
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "demo_rag_doc.md"
    if path.exists():
        return
    if next(iter_documents(str(docs_dir)), None) is not None:
        return                                        # 已有语料：不要往里塞演示文档
    path.write_text(DEMO_DOC, encoding="utf-8")


def print_answer(result: dict) -> None:
    """打印答案 + 引用出处 + 校验信息。

    ⚠ 校验结果一定要显式打印（而不是只写日志）：
       invalid_refs 非空（引用了不存在的编号）或 refs 为空（无据结论），
       这两种情况都比「答案不好看」严重得多。
    """
    answer, packed = result["answer"], result["packed"]
    print(f"支持度（原始余弦）: {result['vec_support']:.4f}"
          f"    上下文预算: {packed['used_tokens']}/{packed['budget']}"
          f"（利用率 {packed['utilization'] * 100:.0f}%）")

    if answer["refused"]:
        print(f"\n[已拒答] 原因码: {answer['refusal_reason']}")
        print(answer["text"])
        if answer.get("raw_text"):
            print(f"（模型原始输出已保留：{answer['raw_text'][:120]}…）")
        return

    print("\n答案：")
    print(answer["text"])
    if answer["citations"]:
        print("\n引用出处：")
        for cite in answer["citations"]:
            heading = f" / {cite['heading']}" if cite.get("heading") else ""
            print(f"  [{cite['ref']}] {cite['source']}{heading}")
    check = answer["validation"]
    print(f"\n引用校验：合法 {len(check['valid_refs'])} 个"
          f"，非法 {check['invalid_refs'] or '无'}"
          f"，未被引用 {check['unused_refs'] or '无'}"
          f"，覆盖率 {check['coverage'] * 100:.0f}%")


def run_eval(pipeline: RAGPipeline, args, parser) -> None:
    """评估子命令。

    两种模式：
        --check  只做锚点自检（零依赖、完全可复现，标注草稿阶段就该先跑）
        默认     检索指标运行器：跑完四组消融配置并出对比表

    刻意不做生成评测：检索指标必须完全离线复现，否则「跑一次评测要花钱」，
    回归门禁就没人跑了。生成侧指标（幻觉率/误拒率）需要真实 LLM，属后续阶段。
    """
    if not args.golden:
        parser.error("eval 需要评测集，例如：--golden eval/golden_set.jsonl")
    try:
        items = load_golden_set(args.golden)
    except (OSError, ValueError) as error:
        raise SystemExit(f"读取评测集失败：{error}") from error

    if not items:
        raise SystemExit(f"评测集为空：{args.golden}")
    # 未核验的样本不进指标分母（见 evaluation 模块说明），这里给出可执行提示
    pending = [i.qid for i in items if not i.reviewed]
    print(f"载入评测集：{len(items)} 条"
          f"（可答 {sum(1 for i in items if i.answerable)} / "
          f"不可答 {sum(1 for i in items if not i.answerable)}）")
    if pending:
        print(f"⚠ {len(pending)} 条未标记 reviewed，不会计入指标：{pending[:5]}")
    if not any(i.answerable and i.reviewed for i in items):
        print("⚠ 没有任何「可答且已核验」的样本，指标分母为 0 —— 先把标注核验完再来跑指标")

    if args.check:
        # 自检需要父块文本与标题，这里就地回填（只建一次索引）
        pipeline.index(str(Path(args.docs)))
        children = pipeline.store.list_children()
        parents = pipeline.store.get_parents([c["parent_id"] for c in children])
        for child in children:
            parent = parents.get(child["parent_id"]) or {}
            child["parent_text"] = parent.get("parent_text")
            child["heading"] = parent.get("heading")
            child["source"] = parent.get("source") or child.get("source")
        unmatched = check_anchors(items, children)
        text = render_check_report(items, unmatched, {"summary": args.docs})
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"\n报告已写入 {args.out}")
        if unmatched:
            # 默认只是"报告"（标注草稿阶段必然有未匹配项，不该视为失败）；
            # --strict 时才当作门禁：CI 里用它，避免"跑了检查但永远绿"的橡皮图章。
            level = logger.error if args.strict else logger.warning
            level("%d 个锚点在索引中找不到匹配%s", len(unmatched),
                  "（--strict 已开启，判定为失败）" if args.strict else "")
            if args.strict:
                raise SystemExit(1)
        return

    # 检索指标运行器：四组消融配置，每组独立建索引（避免 BM25 状态互相污染）
    runs = [ABLATIONS[name] for name in args.runs] if args.runs else list(ABLATIONS.values())
    print(f"开始评测：{len(runs)} 组配置 × {len(items)} 条样本"
          f"（top_k={args.top_k}，语料 {args.docs}）")

    def factory(use_bm25: bool, use_mmr: bool) -> RAGPipeline:
        """每组配置都新建 store + pipeline。

        复用同一个 store 会让后一组带着前一组的索引与 BM25 状态，
        四组结果互相污染（典型表现：`纯向量` 那一组也命中大量关键词结果）。
        embedder 复用是安全的——它是无状态装饰器，只做内容→向量的映射。
        不传 generator：检索指标不调用 LLM。
        """
        return RAGPipeline(pipeline.embedder, type(pipeline.store)(),
                           use_bm25=use_bm25, use_mmr=use_mmr)

    try:
        reports = run_experiment(items, str(Path(args.docs)), factory, runs=runs,
                                 top_k=args.top_k)
    except ValueError as error:
        raise SystemExit(f"评测失败：{error}") from error

    text = render_report(reports)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n报告已写入 {args.out}")


def _open_browser_later(url: str, delay: float = 2.0) -> None:
    """延迟打开浏览器。

    为什么要自己开，而不是让 streamlit 开：
      观测台以 **headless 模式**启动（见 run_serve 的说明）——headless 会跳过
      streamlit 首跑时的"邮箱问卷"。那个问卷会在终端里**阻塞等待输入**，
      对"点一下就能看到窗口"的体验是纯粹的摩擦（用户第一次跑必然撞上）。
      headless 的代价是 streamlit 不再自动开浏览器，所以这里补上。

    用 Timer 而不是启动后再开：`stcli.main()` 是阻塞调用，它返回时服务已经停了。
    """
    def _open() -> None:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:                              # noqa: BLE001
            # 无图形环境（服务器/容器）时打开失败是正常的，不该影响服务本身
            logger.debug("自动打开浏览器失败（不影响使用，请手动访问 %s）", url, exc_info=True)

    threading.Timer(delay, _open).start()


def run_serve(args: "argparse.Namespace") -> None:
    """启动可视化观测台（Streamlit）。

    为什么要绕一层 `streamlit.web.cli`，而不是让用户自己敲 `streamlit run`：
      · 参数（语料目录、端口、是否开浏览器）能跟本项目的 CLI 风格保持一致；
      · 用户只需要记住一个入口 `python -m RAG.main serve`；
      · 观测台依赖 streamlit，而 streamlit **不在核心依赖里**——
        本函数负责给出"装什么、怎么装"的明确提示，而不是抛一个 ImportError 调用栈。
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError as error:
        raise SystemExit(
            "观测台需要 streamlit（核心路径不需要它）：\n"
            "    uv pip install streamlit\n"
            "    # 或（实机上按 pyproject 的 webui 分组安装）\n"
            "    uv sync --extra webui") from error

    app = Path(__file__).resolve().parent / "webui.py"
    log_level = "warning" if args.quiet else ("debug" if args.verbose else "info")
    url = f"http://127.0.0.1:{args.port}"
    # streamlit 的 CLI 直接读 sys.argv，因此这里显式构造它（而不是传参）
    sys.argv = [
        "streamlit", "run", str(app),
        "--server.port", str(args.port),
        # ★ 始终 headless：跳过首跑的邮箱问卷（否则第一次运行会卡在终端等输入）。
        #   代价是不再自动开浏览器，由 _open_browser_later() 补上。
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--logger.level", log_level,
        # `--` 之后的参数会转交给应用脚本本身（本项目的 webui 只认 --docs）
        "--", "--docs", args.docs,
    ]
    print(f"观测台启动中：{url}   （Ctrl+C 停止）")
    if not args.no_browser:
        _open_browser_later(url)
    stcli.main()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="可测试、可持久化的 RAG 示例")
    parser.add_argument("command",
                        choices=["demo", "qa", "init-db", "index", "query", "eval", "serve"])
    parser.add_argument("value", nargs="?", help="index 的目录、query/qa 的问题")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--docs", default="./docs",
                        help="memory 模式下 qa 需要现场建索引的文档目录（默认 ./docs）")
    parser.add_argument("--golden", default=None, help="评测集 JSONL（eval 子命令）")
    parser.add_argument("--check", action="store_true",
                        help="eval：只做锚点自检，不跑检索指标")
    parser.add_argument("--strict", action="store_true",
                        help="eval：自检发现未匹配锚点时以非 0 退出（供 CI 当门禁用）")
    parser.add_argument("--runs", nargs="+", choices=list(ABLATIONS),
                        help=f"eval：只跑指定消融配置（可选 {list(ABLATIONS)}），默认全跑")
    parser.add_argument("--out", default=None, help="eval：报告输出路径")
    parser.add_argument("--provider", choices=["echo", "openai"], default=None,
                        help="生成器：qa 默认 echo（离线）；demo 默认不生成")
    parser.add_argument("--model", default=None, help="生成模型名（--provider openai 时生效）")
    parser.add_argument("--port", type=int, default=8501, help="serve：观测台端口")
    parser.add_argument("--no-browser", action="store_true",
                        help="serve：不自动打开浏览器（服务器/CI 环境用）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出过程日志到 stderr（每个文件的索引、每次召回的明细）")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="只输出 WARNING 及以上日志（适合把结果管道给其他程序）")
    args = parser.parse_args()

    # 日志配置只在应用层做一次：库代码只记录，不配置（见 logging_setup 的约定）
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    logger.debug("启动：command=%s docs=%s", args.command, args.docs)

    # serve 走独立分支：它不需要装配 pipeline（面板内部自己建，且要按开关重建）
    if args.command == "serve":
        run_serve(args)
        return

    settings = Settings.from_env()

    if args.command == "init-db":
        pipeline = build_pipeline(settings)
        if not isinstance(pipeline.store, PgVectorStore):
            raise ValueError("init-db 需要 RAG_BACKEND=pgvector")
        pipeline.store.initialize()
        print("数据库结构初始化完成")
        return

    # demo 默认不装配生成器（纯检索链路）；qa 默认 echo，保证离线可跑通生成层
    provider = args.provider if args.provider else ("echo" if args.command == "qa" else None)
    generator = build_generator(provider, settings, args.model) if provider else None
    pipeline = build_pipeline(settings, generator)

    if args.command == "eval":
        run_eval(pipeline, args, parser)
        return

    if args.command == "demo":
        docs_dir = Path(args.value or "./docs")
        ensure_demo_document(docs_dir)
        stat = pipeline.index(str(docs_dir))
        print(f"索引完成：父块 {stat['parents']}，子块 {stat['children']}，跳过 {stat['skipped']}")
        for question in ("连接池最大连接数是多少", "error code 0x80070005"):
            result = pipeline.query(question, top_k=3, min_sim=0.0)
            print(f"\n问题：{question}")
            print(result["prompt"])
        return

    if args.command == "index":
        if not args.value:
            parser.error("index 需要文档目录，例如：python -m RAG.main index ./docs")
        stat = pipeline.index(args.value)
        print(f"索引完成：父块 {stat['parents']}，子块 {stat['children']}，"
              f"跳过 {stat['skipped']}，删除 {stat['deleted']}")
        return

    if not args.value:
        parser.error(f"{args.command} 需要问题，例如："
                     f"python -m RAG.main {args.command} \"连接池最大连接数是多少\"")

    if args.command == "qa":
        # qa 需要索引。memory 模式不跨进程保存索引，因此这里现场建立，
        # 让离线用户一条命令就能跑通「索引 → 检索 → 生成 → 引用校验」全链路。
        if settings.backend == "memory":
            docs_dir = Path(args.docs)
            ensure_demo_document(docs_dir)
            stat = pipeline.index(str(docs_dir))
            print(f"已建立临时索引：父块 {stat['parents']}，子块 {stat['children']}"
                  f"（来自 {docs_dir}）")
        print(f"\n问题：{args.value}")
        print_answer(pipeline.answer(args.value, top_k=args.top_k))
        return

    if settings.backend == "memory":
        raise ValueError("memory 模式不跨进程保存索引；query 请改用 qa（会自动建索引）或启用 pgvector")
    print(pipeline.query(args.value, top_k=args.top_k)["prompt"])


if __name__ == "__main__":
    main()
