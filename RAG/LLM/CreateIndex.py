#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS 向量索引构建 (CreateIndex)
================================

基于预处理 JSON 构建 FAISS 向量索引，支持两种中间结果:
    - 方案 A（rule）: PreProcessed.py  → processed/*.preprocessed.json
    - 方案 B（llm）:  PreprocessLLM.py → processed/llm/*.preprocessed.llm.json

流程:
    1. 读取预处理 JSON 中的文本块
    2. 调用百炼 text-embedding-v4 批量向量化
    3. L2 归一化 + IndexFlatIP（余弦相似度）
    4. 持久化索引、元数据与配置

用法:
    python CreateIndex.py
    python CreateIndex.py --mode llm
    python CreateIndex.py --input processed/llm/ --output faiss_index/llm/ --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
from openai import OpenAI


# =============================================================================
# 路径与模型配置
# =============================================================================

RAG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREPROCESS_DIR = RAG_ROOT / "processed"
DEFAULT_LLM_PREPROCESS_DIR = RAG_ROOT / "processed" / "llm"
DEFAULT_INDEX_DIR = RAG_ROOT / "faiss_index"
DEFAULT_LLM_INDEX_DIR = RAG_ROOT / "faiss_index" / "llm"

INDEX_FILE = DEFAULT_INDEX_DIR / "knowledge.index"
METADATA_FILE = DEFAULT_INDEX_DIR / "metadata.pkl"
CONFIG_FILE = DEFAULT_INDEX_DIR / "config.json"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 10

API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")
ZSHEV_PATH = Path.home() / ".zshenv"

PREPROCESS_SUFFIX = ".preprocessed.json"
PREPROCESS_LLM_SUFFIX = ".preprocessed.llm.json"


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class IndexChunk:
    """写入 FAISS 索引的单条记录。"""

    chunk_id: int
    text: str
    display_name: str
    source_file: str
    source_pages: List[int]
    char_start: int
    char_end: int
    preprocess_file: str
    title: str = ""
    summary: str = ""


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
# 加载预处理结果
# =============================================================================


def iter_preprocess_files(input_path: Path, mode: str = "rule") -> List[Path]:
    """收集预处理 JSON 文件。mode: rule（方案 A）或 llm（方案 B）。"""
    if mode not in {"rule", "llm"}:
        raise ValueError(f"不支持的 preprocess mode: {mode}")

    if input_path.is_file():
        if not input_path.name.endswith(".json"):
            raise ValueError(f"仅支持 JSON 文件: {input_path}")
        if mode == "rule" and input_path.name.endswith(PREPROCESS_LLM_SUFFIX):
            raise ValueError(
                f"当前为 LLM 预处理文件，请使用 --mode llm 或 PreprocessLLM.py 输出目录"
            )
        if mode == "llm" and not input_path.name.endswith(PREPROCESS_LLM_SUFFIX):
            raise ValueError(
                f"请使用 LLM 预处理文件（{PREPROCESS_LLM_SUFFIX}）: {input_path}"
            )
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if mode == "llm":
        files = sorted(input_path.glob(f"*{PREPROCESS_LLM_SUFFIX}"))
        hint_script = "PreprocessLLM.py"
        hint_suffix = PREPROCESS_LLM_SUFFIX
    else:
        files = sorted(
            path
            for path in input_path.glob(f"*{PREPROCESS_SUFFIX}")
            if not path.name.endswith(PREPROCESS_LLM_SUFFIX)
        )
        hint_script = "PreProcessed.py"
        hint_suffix = PREPROCESS_SUFFIX

    if not files:
        raise FileNotFoundError(
            f"未找到预处理文件 (*{hint_suffix})，请先运行 {hint_script}\n"
            f"查找目录: {input_path}"
        )
    return files


def format_page_label(pages: Sequence[int]) -> str:
    """将页码列表格式化为可读标签。"""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"第 {pages[0]} 页"
    return f"第 {pages[0]}-{pages[-1]} 页"


def load_chunks_from_preprocess(json_path: Path) -> Tuple[List[IndexChunk], dict]:
    """从单个预处理 JSON 加载文本块。"""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    source_file = payload.get("source_file", "")
    display_name = Path(source_file).name if source_file else json_path.stem.replace(".preprocessed", "")

    raw_chunks = payload.get("chunks", [])
    chunks: List[IndexChunk] = []

    for item in raw_chunks:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        source_pages = item.get("source_pages", [])
        if not isinstance(source_pages, list):
            source_pages = []

        page_ints: List[int] = []
        for page in source_pages:
            try:
                page_ints.append(int(page))
            except (TypeError, ValueError):
                continue

        chunks.append(
            IndexChunk(
                chunk_id=int(item.get("chunk_id", len(chunks))),
                text=text,
                display_name=display_name,
                source_file=source_file,
                source_pages=sorted(set(page_ints)),
                char_start=int(item.get("char_start", 0)),
                char_end=int(item.get("char_end", 0)),
                preprocess_file=str(json_path.resolve()),
                title=str(item.get("title", "") or "").strip(),
                summary=str(item.get("summary", "") or "").strip(),
            )
        )

    doc_meta = {
        "preprocess_file": str(json_path.resolve()),
        "source_file": source_file,
        "display_name": display_name,
        "chunk_count": len(chunks),
        "preprocess_mode": payload.get("preprocess_mode", "rule"),
        "chunk_size": payload.get("chunk_size"),
        "chunk_overlap": payload.get("chunk_overlap"),
        "chat_model": payload.get("chat_model"),
    }
    return chunks, doc_meta


def load_all_chunks(preprocess_files: Sequence[Path]) -> Tuple[List[IndexChunk], List[dict]]:
    """合并多个预处理文件的文本块，重新编号 chunk_id。"""
    all_chunks: List[IndexChunk] = []
    doc_metas: List[dict] = []

    for json_path in preprocess_files:
        chunks, doc_meta = load_chunks_from_preprocess(json_path)
        if not chunks:
            print(f"  [跳过] {json_path.name} — 无有效文本块")
            continue

        for chunk in chunks:
            all_chunks.append(
                IndexChunk(
                    chunk_id=len(all_chunks),
                    text=chunk.text,
                    display_name=chunk.display_name,
                    source_file=chunk.source_file,
                    source_pages=chunk.source_pages,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    preprocess_file=chunk.preprocess_file,
                    title=chunk.title,
                    summary=chunk.summary,
                )
            )
        doc_metas.append(doc_meta)
        print(f"  [加载] {json_path.name} -> {len(chunks)} 块")

    return all_chunks, doc_metas


# =============================================================================
# Embedding 与 FAISS
# =============================================================================


def embed_texts(client: OpenAI, texts: Sequence[str]) -> np.ndarray:
    """调用 text-embedding-v4 批量向量化。"""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    all_vectors: List[List[float]] = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
            encoding_format="float",
        )
        batch_vectors = [item.embedding for item in response.data]
        all_vectors.extend(batch_vectors)
        print(f"    已向量化 {min(start + len(batch), total)}/{total} 条")

    return np.array(all_vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """L2 归一化后构建 IndexFlatIP 索引（等价余弦相似度）。"""
    if vectors.size == 0:
        raise ValueError("向量数组为空，无法构建 FAISS 索引。")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def chunk_to_metadata(chunk: IndexChunk) -> dict:
    """将 IndexChunk 转为 metadata.pkl 中的字典（兼容问答模块检索展示）。"""
    page_label = format_page_label(chunk.source_pages)
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "display_name": chunk.display_name,
        "source_file": chunk.source_file,
        "source_pages": chunk.source_pages,
        "page_label": page_label,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "preprocess_file": chunk.preprocess_file,
        "title": chunk.title,
        "summary": chunk.summary,
        # 兼容 RegulatoryAssitant 的定位字段
        "chapter": chunk.title or page_label,
        "section": page_label if chunk.title else "",
        "article": "",
    }


def save_index_bundle(
    index: faiss.Index,
    chunks: Sequence[IndexChunk],
    vectors: np.ndarray,
    doc_metas: Sequence[dict],
    index_dir: Path,
    preprocess_mode: str = "rule",
) -> None:
    """持久化 FAISS 索引、元数据与配置。"""
    index_dir.mkdir(parents=True, exist_ok=True)

    index_file = index_dir / "knowledge.index"
    metadata_file = index_dir / "metadata.pkl"
    config_file = index_dir / "config.json"

    faiss.write_index(index, str(index_file))

    metadata = [chunk_to_metadata(chunk) for chunk in chunks]
    with metadata_file.open("wb") as file:
        pickle.dump(metadata, file)

    source_files = sorted({meta.get("source_file", "") for meta in doc_metas if meta.get("source_file")})
    suffix = PREPROCESS_LLM_SUFFIX if preprocess_mode == "llm" else PREPROCESS_SUFFIX
    config = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "preprocess_mode": preprocess_mode,
        "preprocess_suffix": suffix,
        "index_type": "IndexFlatIP",
        "similarity": "cosine",
        "vector_count": len(chunks),
        "document_count": len(doc_metas),
        "source_files": source_files,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[完成] FAISS 索引: {index_file}")
    print(f"[完成] 元数据文件: {metadata_file}")
    print(f"[完成] 配置文件: {config_file}")
    print(f"[完成] 向量条目数: {vectors.shape[0]}")


# =============================================================================
# 构建主流程
# =============================================================================


def index_exists(index_dir: Path) -> bool:
    """检查索引是否已存在。"""
    return (index_dir / "knowledge.index").exists() and (index_dir / "metadata.pkl").exists()


def build_index(
    client: OpenAI,
    preprocess_files: Sequence[Path],
    index_dir: Path,
    force: bool = False,
    preprocess_mode: str = "rule",
) -> None:
    """执行完整的向量索引构建流程。"""
    if index_exists(index_dir) and not force:
        print("检测到已有索引。若需重建，请加参数 --rebuild")
        return

    print("\n[1/3] 加载预处理文本块...")
    chunks, doc_metas = load_all_chunks(preprocess_files)
    if not chunks:
        raise RuntimeError("没有可用于向量化的文本块。")
    print(f"      合计 {len(chunks)} 块，来自 {len(doc_metas)} 份文档")

    print("\n[2/3] 调用 text-embedding-v4 向量化...")
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(client, texts)

    print("\n[3/3] 构建并保存 FAISS 索引...")
    index = build_faiss_index(vectors)
    save_index_bundle(index, chunks, vectors, doc_metas, index_dir, preprocess_mode)


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAISS 向量索引构建 — 读取 PreProcessed.py 输出 + text-embedding-v4"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rule", "llm"],
        default="rule",
        help="预处理方案: rule=PreProcessed, llm=PreprocessLLM",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="预处理 JSON 文件或目录（默认随 --mode 自动选择）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="FAISS 索引输出目录（默认随 --mode 自动选择）",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制重建索引（覆盖已有文件）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        input_path = DEFAULT_LLM_PREPROCESS_DIR if mode == "llm" else DEFAULT_PREPROCESS_DIR

    if args.output:
        index_dir = Path(args.output).expanduser().resolve()
    else:
        index_dir = DEFAULT_LLM_INDEX_DIR if mode == "llm" else DEFAULT_INDEX_DIR

    try:
        preprocess_files = iter_preprocess_files(input_path, mode=mode)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  FAISS 向量索引构建 (CreateIndex)")
    print("=" * 60)
    print(f"  预处理方案: {mode}")
    print(f"  预处理输入: {input_path}")
    print(f"  索引输出: {index_dir}")
    print(f"  向量模型: {EMBEDDING_MODEL} ({EMBEDDING_DIM} 维)")
    print(f"  预处理文件数: {len(preprocess_files)}")
    print("=" * 60)

    try:
        api_key = load_api_key()
        client = create_client(api_key)
        build_index(
            client=client,
            preprocess_files=preprocess_files,
            index_dir=index_dir,
            force=args.rebuild,
            preprocess_mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n[完成] 索引构建结束。")


if __name__ == "__main__":
    main()
