#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库一键构建 — LLM 方案 (GetKnowledgeLLM)
============================================

汇总 PreprocessLLM.py + CreateIndex.py，一次运行完成:
    PDF 逐页提取 → deepseek-v4-pro 语义切分 → text-embedding-v4 向量化 → FAISS 索引

用法:
    python GetKnowledgeLLM.py
    python GetKnowledgeLLM.py --input data/ --rebuild
    python GetKnowledgeLLM.py --skip-preprocess   # 仅重建索引（已有 processed/llm/）
    python GetKnowledgeLLM.py --dry-run           # 仅展示 LLM 分批计划

等价于依次执行:
    python PreprocessLLM.py --input data/ --output processed/llm/
    python CreateIndex.py --mode llm --input processed/llm/ --output faiss_index/llm/ --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from CreateIndex import (
    DEFAULT_LLM_INDEX_DIR,
    DEFAULT_LLM_PREPROCESS_DIR,
    build_index,
    create_client,
    iter_preprocess_files,
    load_api_key,
)
from PreprocessLLM import (
    BATCH_PAGE_OVERLAP,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    MAX_BATCH_CHARS,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    default_output_file,
    iter_pdf_files,
    preprocess_pdf_llm,
    print_batch_plan,
    save_llm_result,
)

BASE_DIR = Path(__file__).resolve().parent


def run_preprocess_step(
    pdf_files: list[Path],
    output_dir: Path,
    client,
    max_batch_chars: int,
    min_chunk_chars: int,
    max_chunk_chars: int,
    page_overlap: int,
) -> int:
    """执行 LLM 语义预处理，返回成功处理的文档数。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for pdf_path in pdf_files:
        try:
            print(f"\n[LLM 预处理] {pdf_path.name}")
            result = preprocess_pdf_llm(
                pdf_path=pdf_path,
                client=client,
                max_batch_chars=max_batch_chars,
                min_chunk_chars=min_chunk_chars,
                max_chunk_chars=max_chunk_chars,
                page_overlap=page_overlap,
            )
            output_path = default_output_file(pdf_path, output_dir)
            save_llm_result(result, output_path)
            print(
                f"  页数 {result.total_pages} | 有效页 {result.valid_pages} | "
                f"批次数 {result.batch_count} | 文本块 {len(result.chunks)} | "
                f"输出 {output_path.name}"
            )
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [错误] LLM 预处理失败: {exc}", file=sys.stderr)

    return success_count


def run_index_step(
    preprocess_dir: Path,
    index_dir: Path,
    force: bool,
) -> None:
    """执行 FAISS 索引构建（LLM 预处理结果）。"""
    preprocess_files = iter_preprocess_files(preprocess_dir, mode="llm")
    api_key = load_api_key()
    client = create_client(api_key)
    build_index(
        client=client,
        preprocess_files=preprocess_files,
        index_dir=index_dir,
        force=force,
        preprocess_mode="llm",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="知识库一键构建（LLM 方案）— 语义切分 + FAISS 向量索引"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"PDF 文件或目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"LLM 预处理 JSON 输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(DEFAULT_LLM_INDEX_DIR),
        help=f"FAISS 索引输出目录（默认: {DEFAULT_LLM_INDEX_DIR}）",
    )
    parser.add_argument("--max-batch-chars", type=int, default=MAX_BATCH_CHARS, help="LLM 每批最大字符数")
    parser.add_argument("--min-chunk-chars", type=int, default=MIN_CHUNK_CHARS, help="建议最小块长度")
    parser.add_argument("--max-chunk-chars", type=int, default=MAX_CHUNK_CHARS, help="建议最大块长度")
    parser.add_argument("--batch-overlap", type=int, default=BATCH_PAGE_OVERLAP, help="批次间重叠页数")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建 FAISS 索引（预处理仍会覆盖写入 JSON）",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过 LLM 预处理，仅基于已有 processed/llm/ 构建索引",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="仅执行 LLM 预处理，不构建索引",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅展示 LLM 分批计划，不调用 API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_input = Path(args.input).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()

    print("=" * 60)
    print("  知识库一键构建 — LLM 方案 (GetKnowledgeLLM)")
    print("=" * 60)
    print(f"  PDF 输入: {pdf_input}")
    print(f"  预处理输出: {processed_dir}")
    print(f"  索引输出: {index_dir}")
    print(f"  LLM 分批: max_batch_chars={args.max_batch_chars}, overlap={args.batch_overlap} 页")
    print(f"  块长建议: {args.min_chunk_chars}–{args.max_chunk_chars} 字")
    if args.dry_run:
        print("  模式: dry-run（不调用 API）")
    elif args.skip_preprocess:
        print("  模式: 仅构建索引")
    elif args.preprocess_only:
        print("  模式: 仅 LLM 预处理")
    else:
        print("  模式: LLM 预处理 + 构建索引")
    print("=" * 60)

    try:
        if args.dry_run:
            pdf_files = list(iter_pdf_files(pdf_input))
            for pdf_path in pdf_files:
                print_batch_plan(pdf_path, args.max_batch_chars, args.batch_overlap)
            print("\n[完成] dry-run 结束。")
            return

        if not args.skip_preprocess:
            print("\n>>> 步骤 1/2: LLM 语义预处理 (PreprocessLLM)")
            api_key = load_api_key()
            client = create_client(api_key)
            pdf_files = list(iter_pdf_files(pdf_input))
            success = run_preprocess_step(
                pdf_files=pdf_files,
                output_dir=processed_dir,
                client=client,
                max_batch_chars=args.max_batch_chars,
                min_chunk_chars=args.min_chunk_chars,
                max_chunk_chars=args.max_chunk_chars,
                page_overlap=args.batch_overlap,
            )
            if success == 0:
                print("\n[错误] 没有成功预处理的 PDF 文件。", file=sys.stderr)
                sys.exit(1)
            print(f"\n[步骤 1 完成] 成功预处理 {success}/{len(pdf_files)} 份文档")
        else:
            print("\n>>> 跳过步骤 1: 使用已有 LLM 预处理结果")

        if args.preprocess_only:
            print("\n[完成] 仅预处理模式，未构建索引。")
            return

        print("\n>>> 步骤 2/2: FAISS 索引构建 (CreateIndex --mode llm)")
        force_rebuild = args.rebuild or not args.skip_preprocess
        run_index_step(
            preprocess_dir=processed_dir,
            index_dir=index_dir,
            force=force_rebuild,
        )
        print("\n[步骤 2 完成] FAISS 向量库已就绪")

    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n[完成] LLM 知识库构建结束。索引目录:", index_dir)


if __name__ == "__main__":
    main()
