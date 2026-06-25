#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库问答入口 (main)
=========================

在 RAG 根目录运行，支持选择两种知识库模式进行问答:
    - normal: 普通方法（规则切分）→ faiss_index/
    - llm:    LLM 方法（语义切分）  → faiss_index/llm/

每次回答前会先输出检索到的相关向量/文本块信息，再调用 deepseek-v4-pro 生成回答。

用法:
    python main.py
    python main.py --mode llm
    python main.py --mode normal --query "客户经理考核标准是什么？"
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from openai import OpenAI

RAG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RAG_ROOT / "Normal"))

from CreateIndex import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_INDEX_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    create_client,
    embed_texts,
    load_api_key,
)


# =============================================================================
# 模式与模型配置
# =============================================================================

MODE_CONFIG = {
    "normal": {
        "label": "普通方法（规则切分）",
        "index_dir": DEFAULT_INDEX_DIR,
    },
    "llm": {
        "label": "LLM 方法（语义切分）",
        "index_dir": DEFAULT_LLM_INDEX_DIR,
    },
}

CHAT_MODEL = "deepseek-v4-pro"
RETRIEVE_TOP_K = 5
MIN_SCORE = 0.45
MAX_CHUNK_PREVIEW = 120
MAX_CONTEXT_CHARS = 6000
MAX_CHUNK_CHARS = 800
MAX_HISTORY_TURNS = 6


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class RetrievedHit:
    """单条检索命中。"""

    rank: int
    vector_id: int
    score: float
    metadata: dict


@dataclass
class ChatTurn:
    role: str
    content: str


@dataclass
class RAGSession:
    mode: str
    history: List[ChatTurn] = field(default_factory=list)
    last_hits: List[RetrievedHit] = field(default_factory=list)


# =============================================================================
# 索引加载与检索
# =============================================================================


def load_index_bundle(index_dir: Path) -> Tuple[faiss.Index, List[dict], dict]:
    """加载 FAISS 索引、元数据与配置。"""
    index_file = index_dir / "knowledge.index"
    metadata_file = index_dir / "metadata.pkl"
    config_file = index_dir / "config.json"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"未找到索引文件: {index_dir}\n"
            f"请先运行 GetKnowledge.py（normal）或 GetKnowledgeLLM.py（llm）构建向量库。"
        )

    index = faiss.read_index(str(index_file))
    with metadata_file.open("rb") as file:
        metadata = pickle.load(file)

    config: dict = {}
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))

    return index, metadata, config


def search_index(
    client: OpenAI,
    index: faiss.Index,
    metadata: List[dict],
    query: str,
    top_k: int = RETRIEVE_TOP_K,
    min_score: float = MIN_SCORE,
) -> List[RetrievedHit]:
    """向量化问题并在 FAISS 中检索 Top-K。"""
    query_vector = embed_texts(client, [query])
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)
    hits: List[RetrievedHit] = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or float(score) < min_score:
            continue
        if idx >= len(metadata):
            continue
        hits.append(
            RetrievedHit(
                rank=len(hits) + 1,
                vector_id=int(idx),
                score=float(score),
                metadata=metadata[idx],
            )
        )

    # 若全部被分数阈值过滤，保留最高分一条
    if not hits and len(indices[0]) > 0 and indices[0][0] >= 0:
        idx = int(indices[0][0])
        if idx < len(metadata):
            hits.append(
                RetrievedHit(
                    rank=1,
                    vector_id=idx,
                    score=float(scores[0][0]),
                    metadata=metadata[idx],
                )
            )

    return hits


# =============================================================================
# 检索结果展示
# =============================================================================


def format_location(meta: dict) -> str:
    """拼接来源定位信息。"""
    parts = []
    if meta.get("title"):
        parts.append(meta["title"])
    if meta.get("page_label"):
        parts.append(meta["page_label"])
    elif meta.get("source_pages"):
        pages = meta["source_pages"]
        if len(pages) == 1:
            parts.append(f"第 {pages[0]} 页")
        else:
            parts.append(f"第 {pages[0]}-{pages[-1]} 页")
    if meta.get("chapter") and meta["chapter"] not in parts:
        parts.insert(0, meta["chapter"])
    return " | ".join(parts) if parts else "—"


def truncate_text(text: str, max_len: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_retrieved_hits(hits: Sequence[RetrievedHit], mode: str) -> None:
    """在生成回答前，输出与问题相关的向量/文本块。"""
    mode_label = MODE_CONFIG[mode]["label"]
    print("\n" + "=" * 64)
    print(f"  [检索结果] 模式: {mode_label}")
    print("=" * 64)

    if not hits:
        print("  未找到相关向量（将基于模型通用知识回答）。")
        print("=" * 64)
        return

    print(f"  共命中 {len(hits)} 条相关向量:\n")
    for hit in hits:
        meta = hit.metadata
        display_name = meta.get("display_name", "未知来源")
        location = format_location(meta)
        preview = truncate_text(meta.get("text", ""), MAX_CHUNK_PREVIEW)
        summary = meta.get("summary", "")

        print(f"  #{hit.rank}  向量ID={hit.vector_id}  chunk_id={meta.get('chunk_id', '?')}")
        print(f"      相似度: {hit.score:.4f}")
        print(f"      来源: {display_name}")
        if location != "—":
            print(f"      定位: {location}")
        if summary:
            print(f"      摘要: {summary}")
        print(f"      预览: {preview}")
        print()

    print("=" * 64)


# =============================================================================
# 上下文组装与 LLM 生成
# =============================================================================


def build_context_block(hits: Sequence[RetrievedHit]) -> str:
    """将检索结果组装为 LLM 参考上下文。"""
    if not hits:
        return "（未检索到相关文档片段，请基于通用知识谨慎回答，并说明未找到直接依据。）"

    blocks: List[str] = []
    total_chars = 0

    for hit in hits:
        meta = hit.metadata
        display_name = meta.get("display_name", "未知文件")
        location = format_location(meta)
        body = truncate_text(meta.get("text", ""), MAX_CHUNK_CHARS)

        header = f"[引用 {hit.rank}] 向量ID={hit.vector_id} | 来源: {display_name} | 相似度: {hit.score:.3f}"
        if location != "—":
            header += f" | 定位: {location}"

        block = f"{header}\n{body}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def build_system_prompt(mode: str) -> str:
    mode_label = MODE_CONFIG[mode]["label"]
    return (
        f"你是一名严谨的知识库问答助手（当前知识库: {mode_label}）。\n"
        "准确性优先于完整性：宁可少答、拒答，不可错答、臆答。\n"
        "用户已在界面中看到检索到的文档片段，你的回答须遵守：\n"
        "【准确性 — 最高优先级】\n"
        "1. 仅依据参考片段中明确写出的内容作答，不得编造、推测、补充或引申文档未提及的信息；\n"
        "2. 数字、金额、比例、日期、条款编号、文件名称等必须与原文一致，不得改写或约化；\n"
        "3. 多个片段信息不一致或不足以支撑结论时，如实说明「文档中未找到明确规定」或「现有片段无法确定」，"
        "不得自行调和、猜测或合并推断；\n"
        "4. 只回答用户所问，不延伸无关内容；不确定时直接说「无法从文档中确认」，不要给出模糊或或然性表述。\n"
        "【表达风格】\n"
        "5. 用自然、简洁的口语化书面语直接作答，像同事解释一样；\n"
        "6. 不要以「根据提供的参考文档」「根据以上片段」等套话开头；\n"
        "7. 不要在回答末尾单独列出「来源」「章节」「页码」等引用块，不要标注【引用 N】；\n"
        "8. 能一句话说清就用一句话；需要分点时用简短条目，避免过度排版；\n"
        "9. 使用简体中文。"
    )


def build_user_message(query: str, context: str) -> str:
    return (
        "以下是从知识库检索到的参考片段（这是你作答的唯一依据；"
        "用户已看过检索结果，回答时不要重复标注来源）：\n"
        "-----\n"
        f"{context}\n"
        "-----\n\n"
        f"用户问题：{query}\n\n"
        "请严格基于以上片段，用自然简洁的中文直接回答。"
        "片段中没有依据的内容一律不说；没有把握就明确说无法确认，不要猜测。"
    )


def generate_answer(
    client: OpenAI,
    query: str,
    context: str,
    mode: str,
    history: Sequence[ChatTurn],
) -> str:
    """调用 deepseek-v4-pro 流式生成回答。"""
    messages: List[dict] = [{"role": "system", "content": build_system_prompt(mode)}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": build_user_message(query, context)})

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    parts: List[str] = []
    print("\n助手: ", end="", flush=True)
    for chunk in stream:
        content = getattr(chunk.choices[0].delta, "content", None) or ""
        if content:
            print(content, end="", flush=True)
            parts.append(content)
    print("\n")
    return "".join(parts)


# =============================================================================
# RAG 主流程
# =============================================================================


def run_rag_turn(
    client: OpenAI,
    index: faiss.Index,
    metadata: List[dict],
    session: RAGSession,
    query: str,
) -> str:
    """执行一轮 RAG：检索 → 展示向量 → 生成回答。"""
    print(f"\n[检索] 正在查询 {MODE_CONFIG[session.mode]['label']} 向量库...")

    hits = search_index(client, index, metadata, query, top_k=RETRIEVE_TOP_K)
    session.last_hits = hits

    print_retrieved_hits(hits, session.mode)

    context = build_context_block(hits)
    print(f"[生成] 调用 {CHAT_MODEL} 生成回答...\n")
    answer = generate_answer(client, query, context, session.mode, session.history)

    session.history.append(ChatTurn(role="user", content=query))
    session.history.append(ChatTurn(role="assistant", content=answer))
    return answer


# =============================================================================
# 模式选择与交互
# =============================================================================


def choose_mode_interactive() -> str:
    """交互式选择 RAG 模式。"""
    print("\n请选择 RAG 模式:")
    print("  1. 普通方法（规则切分）  → faiss_index/")
    print("  2. LLM 方法（语义切分）  → faiss_index/llm/")
    print("  q. 退出")

    mapping = {"1": "normal", "2": "llm", "normal": "normal", "llm": "llm"}

    while True:
        try:
            choice = input("\n请输入选项 [1/2]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            sys.exit(0)

        if choice in {"q", "quit", "exit", "退出"}:
            print("再见！")
            sys.exit(0)
        if choice in mapping:
            return mapping[choice]
        print("无效选项，请输入 1 或 2。")


def print_welcome(mode: str, config: dict) -> None:
    index_dir = MODE_CONFIG[mode]["index_dir"]
    print("=" * 64)
    print("  RAG 知识库问答")
    print("=" * 64)
    print(f"  模式: {MODE_CONFIG[mode]['label']}")
    print(f"  索引目录: {index_dir}")
    print(f"  向量模型: {config.get('embedding_model', EMBEDDING_MODEL)}")
    print(f"  对话模型: {CHAT_MODEL}")
    print(f"  向量条目: {config.get('vector_count', '未知')} 条")
    if config.get("created_at"):
        print(f"  索引时间: {config['created_at']}")
    print(f"  检索 Top-K: {RETRIEVE_TOP_K}")
    print("-" * 64)
    print("  输入问题开始对话；输入 quit 退出，/refs 查看上一轮检索结果")
    print("=" * 64)


def print_last_hits(session: RAGSession) -> None:
    if not session.last_hits:
        print("\n暂无检索记录，请先提问。")
        return
    print_retrieved_hits(session.last_hits, session.mode)


def run_interactive(
    client: OpenAI,
    index: faiss.Index,
    metadata: List[dict],
    session: RAGSession,
) -> None:
    print("\n已进入对话模式。\n")
    while True:
        try:
            user_input = input("您 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue
        lower = user_input.lower()
        if lower in {"quit", "exit", "q", "退出"}:
            print("\n再见！")
            break
        if lower == "/refs":
            print_last_hits(session)
            continue

        try:
            run_rag_turn(client, index, metadata, session, user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[错误] {exc}")


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 知识库问答 — 支持普通 / LLM 两种模式")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "llm"],
        default="",
        help="知识库模式: normal=规则切分, llm=语义切分（省略则交互选择）",
    )
    parser.add_argument("--query", type=str, default="", help="单次提问（省略则进入交互模式）")
    parser.add_argument("--top-k", type=int, default=RETRIEVE_TOP_K, help="检索返回条数")
    return parser.parse_args()


def main() -> None:
    global RETRIEVE_TOP_K  # noqa: PLW0603
    args = parse_args()
    RETRIEVE_TOP_K = args.top_k

    mode = args.mode or choose_mode_interactive()
    index_dir = MODE_CONFIG[mode]["index_dir"]

    try:
        index, metadata, config = load_index_bundle(index_dir)
        api_key = load_api_key()
        client = create_client(api_key)
        session = RAGSession(mode=mode)

        if args.query:
            print_welcome(mode, config)
            run_rag_turn(client, index, metadata, session, args.query.strip())
            return

        print_welcome(mode, config)
        run_interactive(client, index, metadata, session)

    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
