#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库一键构建 (GetKnowledge)
============================

汇总 PreProcessed.py + CreateIndex.py，一次运行完成:
    PDF 预处理 → text-embedding-v4 向量化 → FAISS 索引持久化

用法:
    python GetKnowledge.py
    python GetKnowledge.py --input data/ --rebuild
    python GetKnowledge.py --skip-preprocess   # 仅重建索引（已有 processed/）

等价于依次执行:
    python PreProcessed.py --input data/ --output processed/
    python CreateIndex.py --input processed/ --output faiss_index/ --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from CreateIndex import (
    DEFAULT_INDEX_DIR,
    DEFAULT_PREPROCESS_DIR,
    build_index,
    create_client,
    iter_preprocess_files,
    load_api_key,
)
from PreProcessed import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    default_output_file,
    iter_pdf_files,
    preprocess_pdf,
    save_preprocess_result,
)


BASE_DIR = Path(__file__).resolve().parent


def run_preprocess_step(
    pdf_files: list[Path],
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """执行 PDF 预处理，返回成功处理的文档数。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for pdf_path in pdf_files:
        try:
            print(f"\n[预处理] {pdf_path.name}")
            result = preprocess_pdf(
                pdf_path=pdf_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            output_path = default_output_file(pdf_path, output_dir)
            save_preprocess_result(result, output_path)
            print(
                f"  页数 {result.total_pages} | 有效页 {result.valid_pages} | "
                f"文本块 {len(result.chunks)} | 输出 {output_path.name}"
            )
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [错误] 预处理失败: {exc}", file=sys.stderr)

    return success_count


def run_index_step(
    preprocess_dir: Path,
    index_dir: Path,
    force: bool,
) -> None:
    """执行 FAISS 索引构建。"""
    preprocess_files = iter_preprocess_files(preprocess_dir)
    api_key = load_api_key()
    client = create_client(api_key)
    build_index(
        client=client,
        preprocess_files=preprocess_files,
        index_dir=index_dir,
        force=force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="知识库一键构建 — PDF 预处理 + FAISS 向量索引"
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
        help=f"预处理 JSON 输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(DEFAULT_INDEX_DIR),
        help=f"FAISS 索引输出目录（默认: {DEFAULT_INDEX_DIR}）",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="文本块重叠")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建 FAISS 索引（预处理仍会覆盖写入 JSON）",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过 PDF 预处理，仅基于已有 processed/ 构建索引",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="仅执行 PDF 预处理，不构建索引",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_input = Path(args.input).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()

    print("=" * 60)
    print("  知识库一键构建 (GetKnowledge)")
    print("=" * 60)
    print(f"  PDF 输入: {pdf_input}")
    print(f"  预处理输出: {processed_dir}")
    print(f"  索引输出: {index_dir}")
    print(f"  分割参数: chunk_size={args.chunk_size}, overlap={args.chunk_overlap}")
    if args.skip_preprocess:
        print("  模式: 仅构建索引")
    elif args.preprocess_only:
        print("  模式: 仅预处理")
    else:
        print("  模式: 预处理 + 构建索引")
    print("=" * 60)

    try:
        if not args.skip_preprocess:
            print("\n>>> 步骤 1/2: PDF 预处理 (PreProcessed)")
            pdf_files = list(iter_pdf_files(pdf_input))
            success = run_preprocess_step(
                pdf_files=pdf_files,
                output_dir=processed_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            if success == 0:
                print("\n[错误] 没有成功预处理的 PDF 文件。", file=sys.stderr)
                sys.exit(1)
            print(f"\n[步骤 1 完成] 成功预处理 {success}/{len(pdf_files)} 份文档")
        else:
            print("\n>>> 跳过步骤 1: 使用已有预处理结果")

        if args.preprocess_only:
            print("\n[完成] 仅预处理模式，未构建索引。")
            return

        print("\n>>> 步骤 2/2: FAISS 索引构建 (CreateIndex)")
        # 刚完成预处理或用户指定 rebuild 时，强制重建索引
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

    print("\n[完成] 知识库构建结束。索引目录:", index_dir)


if __name__ == "__main__":
    main()
