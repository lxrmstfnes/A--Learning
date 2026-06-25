#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 文档预处理 — 方案 B：LLM 语义切分 (PreprocessLLM)
======================================================

标准大模型预处理流程:
    1. pypdf 逐页提取 PDF 文本，记录页码，处理空白/异常页
    2. 按上下文窗口将页面分批（批次间 1 页重叠，避免边界断句）
    3. 调用 deepseek-v4-pro，按语义完整性切分并返回结构化 JSON
    4. 校验 LLM 标注的页码（文本重合推断），修正错误映射

用法:
    python PreprocessLLM.py
    python PreprocessLLM.py --input data/某文件.pdf
    python PreprocessLLM.py --dry-run          # 仅展示分批计划，不调用 API

API Key:
    优先读取环境变量 DASHSCOPE_API_KEY / OPENAI_API_KEY；
    若未设置，则从 ~/.zshenv 解析。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

RAG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_ROOT / "Normal"))

from PreProcessed import (
    PageRecord,
    extract_pages_from_pdf,
    iter_pdf_files,
)


# =============================================================================
# 路径与模型配置
# =============================================================================

DEFAULT_INPUT_DIR = RAG_ROOT / "data"
DEFAULT_OUTPUT_DIR = RAG_ROOT / "processed" / "llm"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHAT_MODEL = "deepseek-v4-pro"

API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
ZSHEV_PATH = Path.home() / ".zshenv"

# 每批送入 LLM 的最大字符数（为 prompt / 输出预留空间）
MAX_BATCH_CHARS = 6000

# LLM 切分目标长度
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1200

# 批次间重叠页数，避免语义段落在批次边界被截断
BATCH_PAGE_OVERLAP = 1

# LLM 调用失败时的最大重试次数
MAX_LLM_RETRIES = 2


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class LLMTextChunk:
    """LLM 语义切分后的文本块。"""

    chunk_id: int
    text: str
    source_pages: List[int]
    title: str = ""
    summary: str = ""
    batch_index: int = 0
    llm_source_pages: List[int] = field(default_factory=list)
    pages_corrected: bool = False


@dataclass
class PreprocessLLMResult:
    """LLM 预处理完整结果。"""

    source_file: str
    preprocess_mode: str
    chat_model: str
    total_pages: int
    valid_pages: int
    empty_pages: List[int]
    error_pages: List[int]
    batch_count: int
    min_chunk_chars: int
    max_chunk_chars: int
    pages: List[PageRecord]
    chunks: List[LLMTextChunk]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# =============================================================================
# API Key 与客户端
# =============================================================================


def load_api_key() -> str:
    """获取百炼 API Key。"""
    for name in API_KEY_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if value:
            return value

    if not ZSHEV_PATH.exists():
        raise EnvironmentError(
            f"未找到 API Key。请在 {ZSHEV_PATH} 中设置 export DASHSCOPE_API_KEY=xxx"
        )

    content = ZSHEV_PATH.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"export\s+(DASHSCOPE_API_KEY|OPENAI_API_KEY)\s*=\s*['\"]?([^'\"\n#]+)['\"]?"
    )
    found: Dict[str, str] = {}
    for env_name, env_value in pattern.findall(content):
        found[env_name] = env_value.strip()

    for name in API_KEY_ENV_NAMES:
        if name in found and found[name]:
            return found[name]

    raise EnvironmentError(
        f"在环境变量和 {ZSHEV_PATH} 中均未找到 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。"
    )


def create_client(api_key: str) -> OpenAI:
    """创建百炼 OpenAI 兼容客户端。"""
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


# =============================================================================
# 2. 页面分批
# =============================================================================


def estimate_page_block_chars(page: PageRecord) -> int:
    """估算单页在 prompt 中的字符占用（含 [PAGE n] 标记）。"""
    return len(page.text) + 32


def group_pages_into_batches(
    pages: Sequence[PageRecord],
    max_chars: int = MAX_BATCH_CHARS,
    page_overlap: int = BATCH_PAGE_OVERLAP,
) -> List[List[PageRecord]]:
    """
    将有效页按字符预算分批，批次间保留 page_overlap 页重叠。

    重叠页有助于 LLM 在批次边界处保持语义段落完整。
    """
    if not pages:
        return []

    batches: List[List[PageRecord]] = []
    start = 0

    while start < len(pages):
        batch: List[PageRecord] = []
        char_count = 0
        index = start

        while index < len(pages):
            page = pages[index]
            page_chars = estimate_page_block_chars(page)
            if batch and char_count + page_chars > max_chars:
                break
            batch.append(page)
            char_count += page_chars
            index += 1

        if not batch:
            batch.append(pages[start])
            index = start + 1

        batches.append(batch)

        if index >= len(pages):
            break

        next_start = index - page_overlap
        if next_start <= start:
            next_start = index
        start = next_start

    return batches


def format_pages_for_prompt(pages: Sequence[PageRecord]) -> Tuple[str, int, int]:
    """将一批页面格式化为带 [PAGE n] 标记的 prompt 文本。"""
    blocks: List[str] = []
    for page in pages:
        blocks.append(f"[PAGE {page.page_number}]\n{page.text}")

    prompt_text = "\n\n".join(blocks)
    return prompt_text, pages[0].page_number, pages[-1].page_number


# =============================================================================
# 3. LLM 语义切分
# =============================================================================


def build_system_prompt(min_chars: int, max_chars: int) -> str:
    """构建 LLM 系统提示词。"""
    return (
        "你是专业的文档结构化预处理助手，负责将 PDF 提取文本切分为适合 RAG 检索的语义段落。\n"
        "必须严格遵守以下规则：\n"
        f"1. 每个段落块长度控制在 {min_chars}–{max_chars} 字左右，优先在章节、条款、段落边界切分；\n"
        "2. text 字段必须逐字来自输入原文，不得改写、总结、补充或编造；\n"
        "3. source_pages 必须是输入中出现的页码整数数组，且与 text 内容对应；\n"
        "4. title 填写该块所属章节/条款标题（若无则留空字符串）；\n"
        "5. summary 用一句话概括该块主旨（≤50 字，不要引入原文没有的信息）；\n"
        "6. 仅输出 JSON 对象，格式如下，不要输出其他文字：\n"
        '{"chunks": [{"text": "...", "source_pages": [1], "title": "...", "summary": "..."}]}'
    )


def build_user_prompt(pages_text: str, start_page: int, end_page: int) -> str:
    """构建用户消息。"""
    return (
        f"以下文档片段来自 PDF 第 {start_page}–{end_page} 页，每页以 [PAGE n] 标记。\n"
        "请按语义完整性切分，并返回 JSON。\n\n"
        f"{pages_text}"
    )


def parse_llm_json(content: str) -> List[dict]:
    """解析 LLM 返回的 JSON，兼容 markdown 代码块包裹。"""
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]

    raise ValueError("LLM 返回 JSON 缺少 chunks 数组。")


def call_llm_segment(
    client: OpenAI,
    pages: Sequence[PageRecord],
    batch_index: int,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> List[dict]:
    """调用 deepseek-v4-pro 对一批页面做语义切分。"""
    pages_text, start_page, end_page = format_pages_for_prompt(pages)
    messages = [
        {"role": "system", "content": build_system_prompt(min_chars, max_chars)},
        {"role": "user", "content": build_user_prompt(pages_text, start_page, end_page)},
    ]

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_LLM_RETRIES + 2):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content or ""
            chunks = parse_llm_json(content)
            validated = validate_llm_chunks(chunks, pages, min_chars, max_chars)
            print(f"  [批次 {batch_index + 1}] 页 {start_page}-{end_page} -> {len(validated)} 块")
            return validated
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  [批次 {batch_index + 1}] 第 {attempt} 次调用失败: {exc}")

    raise RuntimeError(f"LLM 切分失败（批次 {batch_index + 1}）: {last_error}")


def validate_llm_chunks(
    chunks: Sequence[dict],
    pages: Sequence[PageRecord],
    min_chars: int,
    max_chars: int,
) -> List[dict]:
    """校验 LLM 返回的 chunk 字段完整性。"""
    page_numbers = {page.page_number for page in pages}
    validated: List[dict] = []

    for item in chunks:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        if len(text) < min_chars // 2:
            # 过短块仍保留，但可能是标题行；由 LLM 决定
            pass

        if len(text) > max_chars * 1.5:
            print(f"  [警告] 块长度 {len(text)} 超出建议上限 {max_chars}，仍保留。")

        raw_pages = item.get("source_pages", [])
        if not isinstance(raw_pages, list):
            raw_pages = []

        source_pages = []
        for page_no in raw_pages:
            try:
                page_int = int(page_no)
            except (TypeError, ValueError):
                continue
            if page_int in page_numbers:
                source_pages.append(page_int)

        validated.append(
            {
                "text": text,
                "source_pages": source_pages,
                "title": str(item.get("title", "") or "").strip(),
                "summary": str(item.get("summary", "") or "").strip(),
            }
        )

    if not validated:
        raise ValueError("LLM 未返回有效 chunks。")

    return validated


# =============================================================================
# 4. 页码映射校验
# =============================================================================


def normalize_for_match(text: str) -> str:
    """去除空白便于子串匹配。"""
    return re.sub(r"\s+", "", text)


def infer_source_pages(chunk_text: str, pages: Sequence[PageRecord]) -> List[int]:
    """
    通过文本重合推断 chunk 的真实来源页码。

    从 chunk 首尾各取一段子串，在批次各页中查找命中。
    """
    normalized = normalize_for_match(chunk_text)
    if len(normalized) < 10:
        return []

    head = normalized[: min(60, len(normalized))]
    tail = normalized[max(0, len(normalized) - 60) :]

    matched: List[int] = []
    for page in pages:
        page_norm = normalize_for_match(page.text)
        if head in page_norm or tail in page_norm:
            matched.append(page.page_number)
            continue

        # 降级：检查 chunk 中较长的连续子串
        window = min(30, len(normalized))
        for offset in range(0, max(1, len(normalized) - window + 1), 15):
            snippet = normalized[offset : offset + window]
            if len(snippet) >= 15 and snippet in page_norm:
                matched.append(page.page_number)
                break

    return sorted(set(matched))


def reconcile_source_pages(
    chunk_text: str,
    llm_pages: Sequence[int],
    batch_pages: Sequence[PageRecord],
) -> Tuple[List[int], bool]:
    """
    合并 LLM 标注页码与文本推断页码。

    若 LLM 页码与推断结果无交集，以推断结果为准。
    """
    inferred = infer_source_pages(chunk_text, batch_pages)
    llm_list = sorted(set(int(p) for p in llm_pages))

    if not inferred:
        return llm_list, False

    if not llm_list:
        return inferred, True

    overlap = set(llm_list) & set(inferred)
    if overlap:
        merged = sorted(set(llm_list) | set(inferred))
        corrected = merged != llm_list
        return merged, corrected

    # LLM 页码与文本推断完全不一致，信任文本推断
    return inferred, True


# =============================================================================
# 预处理主流程
# =============================================================================


def deduplicate_chunks(chunks: Sequence[LLMTextChunk]) -> List[LLMTextChunk]:
    """去除批次重叠导致的重复 chunk（按文本归一化去重）。"""
    seen: set[str] = set()
    unique: List[LLMTextChunk] = []

    for chunk in chunks:
        key = normalize_for_match(chunk.text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(chunk)

    for index, chunk in enumerate(unique):
        chunk.chunk_id = index
    return unique


def preprocess_pdf_llm(
    pdf_path: Path,
    client: OpenAI,
    max_batch_chars: int = MAX_BATCH_CHARS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    page_overlap: int = BATCH_PAGE_OVERLAP,
) -> PreprocessLLMResult:
    """执行 LLM 预处理完整流程。"""
    pages, total_pages = extract_pages_from_pdf(pdf_path)

    empty_pages = [page.page_number for page in pages if page.is_empty]
    error_pages = [page.page_number for page in pages if page.error]
    valid_page_records = [page for page in pages if not page.is_empty and not page.error]

    batches = group_pages_into_batches(
        valid_page_records,
        max_chars=max_batch_chars,
        page_overlap=page_overlap,
    )

    raw_chunks: List[LLMTextChunk] = []
    chunk_counter = 0

    for batch_index, batch_pages in enumerate(batches):
        llm_items = call_llm_segment(
            client=client,
            pages=batch_pages,
            batch_index=batch_index,
            min_chars=min_chunk_chars,
            max_chars=max_chunk_chars,
        )

        for item in llm_items:
            final_pages, corrected = reconcile_source_pages(
                chunk_text=item["text"],
                llm_pages=item["source_pages"],
                batch_pages=batch_pages,
            )

            raw_chunks.append(
                LLMTextChunk(
                    chunk_id=chunk_counter,
                    text=item["text"],
                    source_pages=final_pages,
                    title=item["title"],
                    summary=item["summary"],
                    batch_index=batch_index,
                    llm_source_pages=item["source_pages"],
                    pages_corrected=corrected,
                )
            )
            chunk_counter += 1

    final_chunks = deduplicate_chunks(raw_chunks)

    return PreprocessLLMResult(
        source_file=str(pdf_path.resolve()),
        preprocess_mode="llm",
        chat_model=CHAT_MODEL,
        total_pages=total_pages,
        valid_pages=len(valid_page_records),
        empty_pages=empty_pages,
        error_pages=error_pages,
        batch_count=len(batches),
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
        pages=pages,
        chunks=final_chunks,
    )


def save_llm_result(result: PreprocessLLMResult, output_path: Path) -> None:
    """保存 LLM 预处理结果为 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_file": result.source_file,
        "preprocess_mode": result.preprocess_mode,
        "chat_model": result.chat_model,
        "total_pages": result.total_pages,
        "valid_pages": result.valid_pages,
        "empty_pages": result.empty_pages,
        "error_pages": result.error_pages,
        "batch_count": result.batch_count,
        "min_chunk_chars": result.min_chunk_chars,
        "max_chunk_chars": result.max_chunk_chars,
        "chunk_count": len(result.chunks),
        "created_at": result.created_at,
        "pages": [asdict(page) for page in result.pages],
        "chunks": [asdict(chunk) for chunk in result.chunks],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_output_file(pdf_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{pdf_path.stem}.preprocessed.llm.json"


def print_batch_plan(pdf_path: Path, max_batch_chars: int, page_overlap: int) -> None:
    """dry-run：仅展示分批计划。"""
    pages, total_pages = extract_pages_from_pdf(pdf_path)
    valid_pages = [page for page in pages if not page.is_empty and not page.error]
    batches = group_pages_into_batches(valid_pages, max_batch_chars, page_overlap)

    print("-" * 60)
    print(f"文件: {pdf_path.name}")
    print(f"总页数: {total_pages} | 有效页: {len(valid_pages)}")
    print(f"分批数: {len(batches)} | 每批上限: {max_batch_chars} 字符 | 重叠: {page_overlap} 页")
    for index, batch in enumerate(batches):
        page_nums = [page.page_number for page in batch]
        chars = sum(len(page.text) for page in batch)
        print(f"  批次 {index + 1}: 页 {page_nums} | 约 {chars} 字符")


def print_summary(result: PreprocessLLMResult, output_path: Path) -> None:
    """打印预处理摘要。"""
    print("-" * 60)
    print(f"文件: {Path(result.source_file).name}")
    print(f"模型: {result.chat_model} | 模式: {result.preprocess_mode}")
    print(f"总页数: {result.total_pages} | 有效页: {result.valid_pages} | 批次数: {result.batch_count}")
    if result.empty_pages:
        print(f"空白页: {result.empty_pages}")
    if result.error_pages:
        print(f"异常页: {result.error_pages}")

    corrected_count = sum(1 for chunk in result.chunks if chunk.pages_corrected)
    print(f"文本块数: {len(result.chunks)} | 页码修正: {corrected_count} 块")

    if result.chunks:
        sample = result.chunks[0]
        preview = sample.text.replace("\n", " ")[:80]
        title = f" | {sample.title}" if sample.title else ""
        print(f"示例块 #0 | 页码 {sample.source_pages}{title} | {preview}...")

    print(f"输出: {output_path}")


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF 文档预处理方案 B — pypdf 逐页提取 + deepseek-v4-pro 语义切分"
    )
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
        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument("--max-batch-chars", type=int, default=MAX_BATCH_CHARS, help="每批最大字符数")
    parser.add_argument("--min-chunk-chars", type=int, default=MIN_CHUNK_CHARS, help="建议最小块长度")
    parser.add_argument("--max-chunk-chars", type=int, default=MAX_CHUNK_CHARS, help="建议最大块长度")
    parser.add_argument("--batch-overlap", type=int, default=BATCH_PAGE_OVERLAP, help="批次间重叠页数")
    parser.add_argument("--dry-run", action="store_true", help="仅展示分批计划，不调用 LLM")
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
    print("  PDF 文档预处理 — 方案 B (PreprocessLLM)")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  模型: {CHAT_MODEL}")
    print(f"  分批: max_batch_chars={args.max_batch_chars}, overlap={args.batch_overlap} 页")
    print(f"  块长: {args.min_chunk_chars}–{args.max_chunk_chars} 字")
    if args.dry_run:
        print("  模式: dry-run（不调用 API）")
    print("=" * 60)

    if args.dry_run:
        for pdf_path in pdf_files:
            print_batch_plan(pdf_path, args.max_batch_chars, args.batch_overlap)
        print("\n[完成] dry-run 结束。")
        return

    try:
        api_key = load_api_key()
        client = create_client(api_key)
    except EnvironmentError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)

    for pdf_path in pdf_files:
        try:
            print(f"\n[处理] {pdf_path.name}")
            result = preprocess_pdf_llm(
                pdf_path=pdf_path,
                client=client,
                max_batch_chars=args.max_batch_chars,
                min_chunk_chars=args.min_chunk_chars,
                max_chunk_chars=args.max_chunk_chars,
                page_overlap=args.batch_overlap,
            )
            output_path = default_output_file(pdf_path, output_dir)
            save_llm_result(result, output_path)
            print_summary(result, output_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] 处理失败 {pdf_path.name}: {exc}", file=sys.stderr)

    print("\n[完成] LLM 预处理结束。")


if __name__ == "__main__":
    main()
