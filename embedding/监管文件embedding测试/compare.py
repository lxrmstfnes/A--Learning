#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监管文件语义检索脚本
====================

功能概述:
    用户输入一段文字，将其向量化后在 FAISS 索引中检索最相似的监管文件片段。
    支持相似度阈值与 Top1/Top2 分差校验，低置信结果会明确提示「请勿直接采信」。

用法:
    python compare.py                                    # 进入交互式检索
    python compare.py "数据安全管理要求"                   # 单次检索
    python compare.py --top-k 10 "理财产品销售"            # 指定返回条数
    python compare.py --min-score 0.75 --min-margin 0.05 "AI投资顾问资质"

阈值说明:
    --min-score   Top1 相似度下限，默认 0.75
    --min-margin  Top1 与 Top2 的分差下限，默认 0.05
    两项同时满足才视为「高置信」；否则提示不确定，需人工核对原文。

前置条件:
    请先运行 python jianguan.py 构建 FAISS 索引。

依赖:
    与 jianguan.py 相同，详见 requirements.txt
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence, Tuple

from jianguan import (
    FAISS_INDEX_FILE,
    create_embedding_client,
    format_location,
    load_api_key,
    load_index_bundle,
    search,
)


# =============================================================================
# 默认参数
# =============================================================================

# 默认返回的相似片段数量
DEFAULT_TOP_K = 5

# 结果正文中最多展示的字符数
CONTENT_PREVIEW_LEN = 300

# Top1 相似度默认下限（基于当前索引实测，仅供参考）
DEFAULT_MIN_SCORE = 0.75

# Top1 与 Top2 分差默认下限（避免多条高分结果难以区分）
DEFAULT_MIN_MARGIN = 0.05


# =============================================================================
# 索引状态检查
# =============================================================================


def ensure_index_ready() -> dict:
    """
    确认 FAISS 索引已构建，并返回索引配置信息。

    返回:
        config 字典，包含 vector_count、embedding_model 等字段
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"未找到 FAISS 索引: {FAISS_INDEX_FILE}\n"
            "请先运行: python jianguan.py"
        )

    _, _, config = load_index_bundle()
    return config


def print_index_info(config: dict, min_score: float, min_margin: float) -> None:
    """打印当前索引的基本信息与阈值配置。"""
    print("=" * 60)
    print("  监管文件语义检索")
    print("=" * 60)
    print(f"  索引文件: {FAISS_INDEX_FILE.name}")
    print(f"  向量模型: {config.get('embedding_model', '未知')}")
    print(f"  向量条数: {config.get('vector_count', '未知')}")
    if config.get("created_at"):
        print(f"  构建时间: {config['created_at']}")
    print(f"  置信阈值: min_score={min_score:.2f}, min_margin={min_margin:.2f}")
    print("=" * 60)


# =============================================================================
# 置信度评估
# =============================================================================


def assess_confidence(
    results: Sequence[Tuple[float, dict]],
    min_score: float,
    min_margin: float,
) -> Tuple[bool, List[str]]:
    """
    评估 Top1 结果是否达到可采信标准。

    判定规则（需同时满足）:
        1. Top1 相似度 >= min_score
        2. Top1 与 Top2 的分差 >= min_margin

    返回:
        (is_confident, reasons)
        is_confident 为 False 时，reasons 列出未达标原因
    """
    if not results:
        return False, ["未检索到任何结果"]

    top1_score = results[0][0]
    reasons: List[str] = []

    if top1_score < min_score:
        reasons.append(
            f"Top1 相似度 {top1_score:.4f} 低于阈值 {min_score:.2f}"
        )

    if len(results) >= 2:
        margin = top1_score - results[1][0]
        if margin < min_margin:
            reasons.append(
                f"Top1 与 Top2 分差 {margin:.4f} 低于阈值 {min_margin:.2f}，"
                "存在多条近似匹配，条号可能混淆"
            )
    else:
        reasons.append("仅返回 1 条结果，无法计算 Top1/Top2 分差")

    return len(reasons) == 0, reasons


def print_confidence_banner(is_confident: bool, reasons: Sequence[str]) -> None:
    """打印置信度结论与风险提示。"""
    print("\n" + "=" * 60)
    if is_confident:
        print("【置信度评估】高置信 — 候选条款较可靠，仍建议人工核对原文与条号")
    else:
        print("【置信度评估】不确定 — 未找到可靠匹配，请勿直接采信")
        for reason in reasons:
            print(f"  · {reason}")
        print("  建议: 调整表述后重试，或直接查阅监管原文")
    print("=" * 60)


# =============================================================================
# 结果展示
# =============================================================================


def format_result_text(text: str, max_len: int = CONTENT_PREVIEW_LEN) -> str:
    """将多行文本压缩为单行预览。"""
    preview = text.replace("\n", " ").strip()
    if len(preview) <= max_len:
        return preview
    return preview[:max_len] + "..."


def print_search_results(
    query: str,
    results: Sequence[Tuple[float, dict]],
    show_full_text: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> None:
    """
    格式化输出检索结果，并给出置信度评估。

    参数:
        query: 用户输入的检索文本
        results: search() 返回的 (相似度, 元数据) 列表
        show_full_text: 是否输出完整正文（默认只展示预览）
        min_score: Top1 相似度下限
        min_margin: Top1 与 Top2 分差下限
    """
    print(f"\n检索内容: {query}")
    print(f"命中条数: {len(results)}")

    if not results:
        print("未找到相似内容。请尝试换一种表述，或先运行 jianguan.py 重建索引。")
        return

    is_confident, reasons = assess_confidence(results, min_score, min_margin)
    print_confidence_banner(is_confident, reasons)

    for rank, (score, item) in enumerate(results, start=1):
        print("\n" + "-" * 60)

        # 仅对 Top1 标注是否达到阈值（便于快速识别）
        if rank == 1:
            badge = "✓ 达标" if is_confident else "⚠ 未达标"
            print(f"Top {rank} | 相似度: {score:.4f} | {badge}")
        else:
            print(f"Top {rank} | 相似度: {score:.4f}")

        print(f"来源文件: {item.get('display_name', '')}")
        print(f"来源类型: {item.get('source_type', '')}")

        location = format_location(item)
        if location:
            print(f"定位: {location}")
        elif item.get("chapter"):
            print(f"章节: {item['chapter']}")
        if item.get("summary"):
            print(f"概述: {item['summary']}")

        content = item.get("text", "")
        if show_full_text:
            print(f"正文:\n{content}")
        else:
            print(f"正文预览: {format_result_text(content)}")


# =============================================================================
# 检索逻辑
# =============================================================================


def run_query(
    client,
    query: str,
    top_k: int,
    show_full_text: bool,
    min_score: float,
    min_margin: float,
) -> None:
    """
    执行一次检索并打印结果。

    参数:
        client: 百炼 Embedding 客户端
        query: 用户输入文本
        top_k: 返回条数（至少为 2，以便计算分差）
        show_full_text: 是否展示完整正文
        min_score: Top1 相似度下限
        min_margin: Top1 与 Top2 分差下限
    """
    query = query.strip()
    if not query:
        print("输入不能为空，请重新输入。")
        return

    # 至少取 2 条，否则无法评估 Top1/Top2 分差
    fetch_k = max(top_k, 2)
    results = search(client, query, top_k=fetch_k)
    print_search_results(
        query,
        results[:top_k],
        show_full_text=show_full_text,
        min_score=min_score,
        min_margin=min_margin,
    )


def run_interactive(
    client,
    top_k: int,
    show_full_text: bool,
    min_score: float,
    min_margin: float,
) -> None:
    """
    交互式检索模式。

    用户可持续输入问题；输入 quit / exit / 退出 结束程序。
    """
    print("\n已进入交互模式。直接输入问题即可检索，输入 quit 退出。")
    print("示例: 数据安全管理要求 / 人工智能算法监管 / 理财产品销售")
    print(f"当前阈值: min_score={min_score:.2f}, min_margin={min_margin:.2f}\n")

    while True:
        try:
            query = input("请输入检索内容 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q", "退出"}:
            print("已退出。")
            break

        try:
            run_query(client, query, top_k, show_full_text, min_score, min_margin)
        except Exception as exc:  # noqa: BLE001 - 单次检索失败不终止交互
            print(f"[检索失败] {exc}")


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="监管文件 FAISS 语义检索工具")
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="检索文本；省略则进入交互模式",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"返回相似片段数量（默认 {DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Top1 相似度下限，低于此值视为不确定（默认 {DEFAULT_MIN_SCORE}）",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help=f"Top1 与 Top2 分差下限，低于此值视为不确定（默认 {DEFAULT_MIN_MARGIN}）",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="输出完整正文，而非预览",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""
    args = parse_args()

    if args.min_score < 0 or args.min_score > 1:
        print("[错误] --min-score 必须在 0 到 1 之间", file=sys.stderr)
        sys.exit(1)
    if args.min_margin < 0:
        print("[错误] --min-margin 不能为负数", file=sys.stderr)
        sys.exit(1)

    try:
        config = ensure_index_ready()
        print_index_info(config, args.min_score, args.min_margin)

        api_key = load_api_key()
        client = create_embedding_client(api_key)

        if args.query:
            run_query(
                client,
                args.query,
                args.top_k,
                args.full,
                args.min_score,
                args.min_margin,
            )
            return

        run_interactive(
            client,
            args.top_k,
            args.full,
            args.min_score,
            args.min_margin,
        )
    except Exception as exc:  # noqa: BLE001 - 统一输出错误信息
        print(f"\n[错误] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
