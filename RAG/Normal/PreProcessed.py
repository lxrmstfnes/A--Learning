#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 文档预处理 (PreProcessed)
=============================

预处理流程:
    1. 逐页提取 PDF 文本，记录页码，处理空白/异常页
    2. 清洗与规范化提取文本
    3. 递归字符分割（chunk_size=1000, overlap=200）
    4. 基于字符偏移，建立文本块与来源页码的映射

用法:
    python PreProcessed.py
    python PreProcessed.py --input data/某文件.pdf
    python PreProcessed.py --input data/ --output processed/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from pypdf import PdfReader


# =============================================================================
# 路径与默认参数
# =============================================================================

RAG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = RAG_ROOT / "data"
DEFAULT_OUTPUT_DIR = RAG_ROOT / "processed"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PAGE_SEPARATOR = "\n\n"

# 递归分割优先级：段落 -> 行 -> 中文句读 -> 英文句点 -> 空格 -> 字符
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class PageRecord:
    """单页 PDF 提取结果。"""

    page_number: int
    text: str
    is_empty: bool = False
    error: Optional[str] = None


@dataclass
class PageSpan:
    """全文中某一页文本对应的字符区间（左闭右开）。"""

    page_number: int
    char_start: int
    char_end: int


@dataclass
class TextChunk:
    """分割后的文本块及页码映射。"""

    chunk_id: int
    text: str
    char_start: int
    char_end: int
    source_pages: List[int] = field(default_factory=list)


@dataclass
class PreprocessResult:
    """单份 PDF 的完整预处理结果。"""

    source_file: str
    total_pages: int
    valid_pages: int
    empty_pages: List[int]
    error_pages: List[int]
    chunk_size: int
    chunk_overlap: int
    full_text_length: int
    pages: List[PageRecord]
    chunks: List[TextChunk]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# =============================================================================
# 1. 逐页 PDF 提取
# =============================================================================


def normalize_page_text(raw: Optional[str]) -> str:
    """清洗单页文本：统一换行、去除首尾空白、压缩连续空行。"""
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_from_pdf(pdf_path: Path) -> Tuple[List[PageRecord], int]:
    """
    逐页提取 PDF 文本并记录页码。

    返回:
        (页面记录列表, PDF 总页数)
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    records: List[PageRecord] = []

    for index in range(total_pages):
        page_number = index + 1
        try:
            raw_text = reader.pages[index].extract_text()
            text = normalize_page_text(raw_text)
            is_empty = len(text) == 0

            records.append(
                PageRecord(
                    page_number=page_number,
                    text=text,
                    is_empty=is_empty,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 单页失败不中断整份文档
            records.append(
                PageRecord(
                    page_number=page_number,
                    text="",
                    is_empty=True,
                    error=str(exc),
                )
            )

    return records, total_pages


# =============================================================================
# 2. 合并有效页文本
# =============================================================================


def build_full_text(pages: Sequence[PageRecord]) -> Tuple[str, List[PageSpan], List[int], List[int]]:
    """
    将非空页拼接为全文，并记录每页字符区间。

    返回:
        (full_text, page_spans, empty_pages, error_pages)
    """
    full_text = ""
    page_spans: List[PageSpan] = []
    empty_pages: List[int] = []
    error_pages: List[int] = []

    for page in pages:
        if page.error:
            error_pages.append(page.page_number)
        if page.is_empty:
            empty_pages.append(page.page_number)
            continue

        if full_text:
            full_text += PAGE_SEPARATOR

        char_start = len(full_text)
        full_text += page.text
        page_spans.append(
            PageSpan(
                page_number=page.page_number,
                char_start=char_start,
                char_end=len(full_text),
            )
        )

    return full_text, page_spans, empty_pages, error_pages


# =============================================================================
# 3. 递归字符分割器
# =============================================================================


class RecursiveCharacterTextSplitter:
    """
    递归字符文本分割器。

    优先在语义边界（段落、行、句读符）处切分，尽量保持 chunk_size 以内。
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators or DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> List[str]:
        """将文本分割为多个 chunk 字符串。"""
        if not text:
            return []

        splits = self._split_text(text, self.separators)
        return self._merge_splits(splits)

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """分割文本并返回 (chunk_text, char_start, char_end)。"""
        chunks = self.split_text(text)
        if not chunks:
            return []

        result: List[Tuple[str, int, int]] = []
        search_from = 0

        for chunk in chunks:
            start = text.find(chunk, search_from)
            if start < 0:
                start = text.find(chunk)
            if start < 0:
                raise RuntimeError("无法在原文中定位 chunk 偏移，请检查分割逻辑。")

            end = start + len(chunk)
            result.append((chunk, start, end))
            search_from = max(start + 1, end - self.chunk_overlap)

        return result

    def _split_text(self, text: str, separators: Sequence[str]) -> List[str]:
        """递归按分隔符优先级切分过长片段。"""
        final_chunks: List[str] = []
        separator = separators[-1]
        next_separators: List[str] = []

        for index, candidate in enumerate(separators):
            if candidate == "":
                separator = candidate
                break
            if candidate in text:
                separator = candidate
                next_separators = list(separators[index + 1 :])
                break

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        merged: List[str] = []
        for index, split in enumerate(splits):
            if split:
                merged.append(split)
            if separator and index < len(splits) - 1:
                merged.append(separator)

        good_splits: List[str] = []
        for split in merged:
            if not split:
                continue
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                if not next_separators:
                    good_splits.extend(self._hard_split(split))
                else:
                    good_splits.extend(self._split_text(split, next_separators))

        return good_splits

    def _hard_split(self, text: str) -> List[str]:
        """无合适分隔符时按固定长度硬切。"""
        return [
            text[index : index + self.chunk_size]
            for index in range(0, len(text), self.chunk_size)
        ]

    def _merge_splits(self, splits: Sequence[str]) -> List[str]:
        """将细粒度 split 合并为带 overlap 的 chunk。"""
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current and current_len + split_len > self.chunk_size:
                chunk_text = "".join(current)
                if chunk_text:
                    chunks.append(chunk_text)

                while current and current_len > self.chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed)

                while current and current_len + split_len > self.chunk_size:
                    removed = current.pop(0)
                    current_len -= len(removed)

            current.append(split)
            current_len += split_len

        if current:
            chunks.append("".join(current))

        return chunks


# =============================================================================
# 4. 文本块与页码映射
# =============================================================================


def map_chunk_to_pages(char_start: int, char_end: int, page_spans: Sequence[PageSpan]) -> List[int]:
    """
    根据字符区间 [char_start, char_end) 匹配来源页码。

    任一 page span 与 chunk 区间有交集，即视为来源页。
    """
    pages: List[int] = []
    for span in page_spans:
        if span.char_start < char_end and span.char_end > char_start:
            pages.append(span.page_number)
    return pages


def split_and_map_pages(
    full_text: str,
    page_spans: Sequence[PageSpan],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[TextChunk]:
    """分割全文并为每个 chunk 建立页码映射。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_results = splitter.split_text_with_offsets(full_text)

    chunks: List[TextChunk] = []
    for chunk_id, (text, char_start, char_end) in enumerate(split_results):
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=text,
                char_start=char_start,
                char_end=char_end,
                source_pages=map_chunk_to_pages(char_start, char_end, page_spans),
            )
        )
    return chunks


# =============================================================================
# 预处理主流程
# =============================================================================


def preprocess_pdf(
    pdf_path: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> PreprocessResult:
    """执行单份 PDF 的完整预处理流程。"""
    pages, total_pages = extract_pages_from_pdf(pdf_path)
    full_text, page_spans, empty_pages, error_pages = build_full_text(pages)

    chunks = split_and_map_pages(
        full_text=full_text,
        page_spans=page_spans,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return PreprocessResult(
        source_file=str(pdf_path.resolve()),
        total_pages=total_pages,
        valid_pages=len(page_spans),
        empty_pages=empty_pages,
        error_pages=error_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        full_text_length=len(full_text),
        pages=pages,
        chunks=chunks,
    )


def save_preprocess_result(result: PreprocessResult, output_path: Path) -> None:
    """将预处理结果保存为 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_file": result.source_file,
        "total_pages": result.total_pages,
        "valid_pages": result.valid_pages,
        "empty_pages": result.empty_pages,
        "error_pages": result.error_pages,
        "chunk_size": result.chunk_size,
        "chunk_overlap": result.chunk_overlap,
        "full_text_length": result.full_text_length,
        "chunk_count": len(result.chunks),
        "created_at": result.created_at,
        "pages": [asdict(page) for page in result.pages],
        "chunks": [asdict(chunk) for chunk in result.chunks],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_pdf_files(input_path: Path) -> Iterable[Path]:
    """解析输入路径，返回待处理的 PDF 文件列表。"""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持 PDF 文件: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"目录中未找到 PDF 文件: {input_path}")
    return pdfs


def default_output_file(pdf_path: Path, output_dir: Path) -> Path:
    """根据 PDF 文件名生成输出 JSON 路径。"""
    return output_dir / f"{pdf_path.stem}.preprocessed.json"


def print_summary(result: PreprocessResult, output_path: Path) -> None:
    """打印单份文档预处理摘要。"""
    print("-" * 60)
    print(f"文件: {Path(result.source_file).name}")
    print(f"总页数: {result.total_pages} | 有效页: {result.valid_pages}")
    if result.empty_pages:
        print(f"空白页: {result.empty_pages}")
    if result.error_pages:
        print(f"异常页: {result.error_pages}")
    print(f"全文长度: {result.full_text_length} 字符")
    print(f"文本块数: {len(result.chunks)} (size={result.chunk_size}, overlap={result.chunk_overlap})")

    if result.chunks:
        sample = result.chunks[0]
        preview = sample.text.replace("\n", " ")[:80]
        print(f"示例块 #0 | 页码 {sample.source_pages} | {preview}...")

    print(f"输出: {output_path}")


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF 文档预处理 — pypdf 提取 + 递归字符分割 + 页码映射")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"PDF 文件或目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"预处理结果输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="文本块重叠长度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    try:
        pdf_files = iter_pdf_files(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  PDF 文档预处理 (PreProcessed)")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  分割参数: chunk_size={args.chunk_size}, overlap={args.chunk_overlap}")
    print("=" * 60)

    for pdf_path in pdf_files:
        try:
            result = preprocess_pdf(
                pdf_path=pdf_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            output_path = default_output_file(pdf_path, output_dir)
            save_preprocess_result(result, output_path)
            print_summary(result, output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] 处理失败 {pdf_path.name}: {exc}", file=sys.stderr)

    print("\n[完成] 预处理结束。")


if __name__ == "__main__":
    main()
