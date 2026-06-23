#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监管文件 RAG 命令行助手 (Regulatory Assistant)
==============================================

功能概述:
    基于已构建的 FAISS 向量索引，提供命令行交互式问答。
    每次回复前自动执行「检索增强生成 (RAG)」流程:

        用户提问
          -> text-embedding-v4 向量化
          -> FAISS 粗召回
          -> 查询后处理 + 重排序
          -> 组装监管上下文
          -> deepseek-v4-pro 生成回答
          -> 命令行展示

用法:
    python RegulatoryAssitant.py              # 进入交互对话
    python RegulatoryAssitant.py --query "理财产品销售有哪些监管要求？"

API Key:
    优先读取环境变量 DASHSCOPE_API_KEY / OPENAI_API_KEY；
    若未设置，则从 ~/.zshenv 解析。

前置条件:
    请先运行 embedding/监管文件embedding测试/jianguan.py 构建 FAISS 索引。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from openai import OpenAI


# =============================================================================
# 路径与模型配置
# =============================================================================

# 当前脚本所在目录（RAG/）
BASE_DIR = Path(__file__).resolve().parent

# 项目根目录
PROJECT_ROOT = BASE_DIR.parent

# FAISS 索引目录（由 jianguan.py 构建）
FAISS_INDEX_DIR = PROJECT_ROOT / "embedding" / "监管文件embedding测试" / "faiss_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "jianguan.index"
METADATA_FILE = FAISS_INDEX_DIR / "metadata.pkl"
CONFIG_FILE = FAISS_INDEX_DIR / "config.json"

# 百炼 OpenAI 兼容接口地址
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Embedding 模型（与索引构建时保持一致）
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024

# 对话生成模型
CHAT_MODEL = "deepseek-v4-pro"

# zshrc / zshenv 中可能存放 API Key 的变量名（按优先级排列）
API_KEY_ENV_NAMES = ("DASHSCOPE_API_KEY", "OPENAI_API_KEY")

# API Key 配置文件路径（用户指定 ~/.zshenv）
ZSHEV_PATH = Path.home() / ".zshenv"


# =============================================================================
# 检索与重排序参数
# =============================================================================

# FAISS 粗召回条数（重排序前）
RETRIEVE_TOP_N = 20

# 重排序后送入 LLM 的上下文条数
RERANK_TOP_K = 5

# 向量相似度下限，低于此值的候选在重排序阶段剔除
MIN_RETRIEVAL_SCORE = 0.55

# 重排序综合分权重：向量分 vs 关键词重合分
RERANK_VECTOR_WEIGHT = 0.70
RERANK_KEYWORD_WEIGHT = 0.30

# MMR 多样性惩罚系数（越大越倾向选取与已选结果差异大的片段）
MMR_LAMBDA = 0.65

# 单条上下文在 prompt 中的最大字符数
MAX_CHUNK_CHARS = 800

# 送入 LLM 的上下文总字符上限（防止超出模型窗口）
MAX_CONTEXT_CHARS = 6000

# 多轮对话保留的历史轮数（不含 system）
MAX_HISTORY_TURNS = 6


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class RetrievedChunk:
    """检索并重排序后的单个上下文片段。"""

    score: float
    vector_score: float
    keyword_score: float
    metadata: dict
    rank: int = 0


@dataclass
class ChatTurn:
    """一轮对话记录。"""

    role: str
    content: str


@dataclass
class RAGSession:
    """RAG 会话状态。"""

    history: List[ChatTurn] = field(default_factory=list)
    last_retrieved: List[RetrievedChunk] = field(default_factory=list)


# =============================================================================
# API Key 与客户端
# =============================================================================


def load_api_key() -> str:
    """
    获取百炼 API Key。

    读取顺序:
        1. 环境变量 DASHSCOPE_API_KEY
        2. 环境变量 OPENAI_API_KEY
        3. ~/.zshenv 中的 export 语句
    """
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
    """创建百炼 OpenAI 兼容客户端（Embedding 与 Chat 共用）。"""
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


# =============================================================================
# FAISS 索引加载
# =============================================================================


def ensure_index_ready() -> dict:
    """
    确认 FAISS 索引文件存在，并返回索引配置。

    返回:
        config 字典，包含 vector_count、embedding_model 等字段
    """
    if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"未找到 FAISS 索引: {FAISS_INDEX_FILE}\n"
            "请先运行: python embedding/监管文件embedding测试/jianguan.py"
        )

    config: dict = {}
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return config


def load_index_bundle() -> Tuple[faiss.Index, List[dict], dict]:
    """加载 FAISS 索引、元数据与配置。"""
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with METADATA_FILE.open("rb") as file:
        metadata = pickle.load(file)

    config = {}
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    return index, metadata, config


# =============================================================================
# Embedding 向量化
# =============================================================================


def embed_texts(client: OpenAI, texts: Sequence[str]) -> np.ndarray:
    """
    调用百炼 text-embedding-v4 生成向量。

    返回:
        shape = (n, EMBEDDING_DIM) 的 float32 数组
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=list(texts),
        dimensions=EMBEDDING_DIM,
        encoding_format="float",
    )
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    return vectors


# =============================================================================
# FAISS 粗召回
# =============================================================================


def faiss_search(
    client: OpenAI,
    query: str,
    top_n: int = RETRIEVE_TOP_N,
) -> List[Tuple[float, dict]]:
    """
    将用户问题向量化，在 FAISS 索引中执行余弦相似度检索。

    返回:
        [(相似度, 元数据), ...] 按相似度降序
    """
    index, metadata, _ = load_index_bundle()
    query_vector = embed_texts(client, [query])
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_n)
    results: List[Tuple[float, dict]] = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((float(score), metadata[idx]))

    return results


# =============================================================================
# 重排序（Rerank）
# =============================================================================


def tokenize_for_overlap(text: str) -> List[str]:
    """
    提取用于关键词重合度计算的中文词块与英文单词。

    说明:
        不依赖外部分词库，使用连续中文 2 字及以上 + 英文单词的简单策略。
    """
    cn_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    en_tokens = re.findall(r"[A-Za-z]{2,}", text.lower())
    return cn_tokens + en_tokens


def keyword_overlap_score(query: str, text: str) -> float:
    """
    计算查询与候选文本的关键词重合度（0~1）。

    算法:
        命中 query 中关键词在 text 中出现的比例，并对长 query 做归一化。
    """
    query_tokens = tokenize_for_overlap(query)
    if not query_tokens:
        return 0.0

    text_lower = text.lower()
    hits = sum(1 for token in query_tokens if token in text or token.lower() in text_lower)
    return hits / len(query_tokens)


def chunk_dedup_key(item: dict) -> str:
    """
    生成去重键：同一文件 + 同一条款只保留最高分候选。

    避免上下文被同一法规的重复片段占满。
    """
    display_name = item.get("display_name", "")
    chapter = item.get("chapter", "")
    section = item.get("section", "")
    article = item.get("article", "")
    return f"{display_name}|{chapter}|{section}|{article}"


def combined_score(vector_score: float, keyword_score: float) -> float:
    """综合重排序分数 = 向量分 * 权重 + 关键词分 * 权重。"""
    return (
        RERANK_VECTOR_WEIGHT * vector_score
        + RERANK_KEYWORD_WEIGHT * keyword_score
    )


def text_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    基于字符 n-gram 集合的 Jaccard 相似度，用于 MMR 多样性惩罚。

    说明:
        避免选取内容高度重叠的相邻片段。
    """
    def char_set(text: str) -> set:
        text = re.sub(r"\s+", "", text)
        if len(text) < 2:
            return {text} if text else set()
        return {text[i : i + 2] for i in range(len(text) - 1)}

    set_a = char_set(text_a)
    set_b = char_set(text_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def mmr_select(
    candidates: Sequence[RetrievedChunk],
    top_k: int,
    lambda_param: float = MMR_LAMBDA,
) -> List[RetrievedChunk]:
    """
    最大边际相关性 (MMR) 重排序，在相关性与多样性之间取得平衡。

    公式:
        MMR = argmax [ λ * Score(d) - (1-λ) * max Sim(d, s) ]
        其中 s 为已选集合中的文档。
    """
    if not candidates:
        return []

    remaining = list(candidates)
    selected: List[RetrievedChunk] = []

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_mmr = float("-inf")

        for idx, candidate in enumerate(remaining):
            relevance = candidate.score
            if selected:
                max_sim = max(
                    text_jaccard_similarity(candidate.metadata.get("text", ""), s.metadata.get("text", ""))
                    for s in selected
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    for rank, chunk in enumerate(selected, start=1):
        chunk.rank = rank
    return selected


def rerank_results(
    query: str,
    raw_results: Sequence[Tuple[float, dict]],
    top_k: int = RERANK_TOP_K,
    min_score: float = MIN_RETRIEVAL_SCORE,
) -> List[RetrievedChunk]:
    """
    对 FAISS 粗召回结果进行查询后处理与重排序。

    处理流程:
        1. 分数过滤 — 剔除低于 min_score 的候选
        2. 去重 — 同一文件+条款保留最高分
        3. 综合打分 — 向量相似度 + 关键词重合度
        4. MMR 多样性选择 — 选取最终 top_k 条
    """
    # 第一步：分数过滤
    filtered = [(score, meta) for score, meta in raw_results if score >= min_score]
    if not filtered:
        # 若全部被过滤，放宽条件保留原始 Top1，避免无上下文可用
        if raw_results:
            filtered = [raw_results[0]]
        else:
            return []

    # 第二步：去重（保留同键最高分）
    best_by_key: Dict[str, Tuple[float, dict]] = {}
    for score, meta in filtered:
        key = chunk_dedup_key(meta)
        if key not in best_by_key or score > best_by_key[key][0]:
            best_by_key[key] = (score, meta)

    # 第三步：综合打分
    candidates: List[RetrievedChunk] = []
    for vector_score, meta in best_by_key.values():
        kw_score = keyword_overlap_score(query, meta.get("text", ""))
        candidates.append(
            RetrievedChunk(
                score=combined_score(vector_score, kw_score),
                vector_score=vector_score,
                keyword_score=kw_score,
                metadata=meta,
            )
        )

    # 按综合分降序排列后进入 MMR
    candidates.sort(key=lambda item: item.score, reverse=True)
    return mmr_select(candidates, top_k=top_k)


# =============================================================================
# 上下文组装
# =============================================================================


def format_location(item: dict) -> str:
    """拼接章/节/条定位信息。"""
    parts = []
    if item.get("chapter"):
        parts.append(item["chapter"])
    if item.get("section"):
        parts.append(item["section"])
    if item.get("article"):
        parts.append(item["article"])
    return " / ".join(parts)


def truncate_text(text: str, max_len: int) -> str:
    """截断过长文本并追加省略号。"""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """
    将重排序后的片段组装为 LLM 可读的参考上下文。

    每条引用包含: 序号、来源文件、定位、相似度、正文摘要。
    """
    if not chunks:
        return "（未检索到相关监管文件片段，请基于通用知识谨慎回答，并明确说明未找到直接依据。）"

    blocks: List[str] = []
    total_chars = 0

    for chunk in chunks:
        item = chunk.metadata
        display_name = item.get("display_name", "未知文件")
        location = format_location(item)
        body = truncate_text(item.get("text", ""), MAX_CHUNK_CHARS)

        header = f"[引用 {chunk.rank}] 来源: {display_name}"
        if location:
            header += f" | 定位: {location}"
        header += f" | 相关度: {chunk.score:.3f}"

        block = f"{header}\n{body}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """构建系统提示词，约束助手行为与回答格式。"""
    return (
        "你是一名专业的金融监管合规助手，专门解答与中国银行业、理财、数据安全等监管政策相关的问题。\n"
        "请严格依据用户提供的「参考监管文件片段」进行回答，遵循以下规则：\n"
        "1. 优先引用参考片段中的具体条款内容，并注明来源文件与章/节/条定位；\n"
        "2. 若参考片段不足以完整回答问题，请明确说明哪些部分缺乏直接依据，不要编造条款；\n"
        "3. 回答使用简体中文，结构清晰，必要时使用条目列举；\n"
        "4. 涉及合规建议时，提醒用户以最新正式发布的监管原文为准。"
    )


def build_user_message_with_context(query: str, context: str) -> str:
    """
    将用户问题与检索上下文组装为单条 user 消息。

    说明:
        采用「上下文 + 问题」的标准 RAG prompt 结构。
    """
    return (
        "以下是从监管文件向量库中检索到的参考片段：\n"
        "-----\n"
        f"{context}\n"
        "-----\n\n"
        f"用户问题：{query}\n\n"
        "请基于以上参考片段回答用户问题。"
    )


# =============================================================================
# LLM 对话生成
# =============================================================================


def generate_answer(
    client: OpenAI,
    query: str,
    context: str,
    history: Sequence[ChatTurn],
    stream: bool = True,
) -> str:
    """
    调用百炼 deepseek-v4-pro 生成回答。

    参数:
        client: OpenAI 兼容客户端
        query: 当前用户问题
        context: 检索组装后的参考上下文
        history: 多轮对话历史
        stream: 是否流式输出到命令行

    返回:
        完整的助手回复文本
    """
    messages: List[dict] = [{"role": "system", "content": build_system_prompt()}]

    # 注入有限历史（不含当前轮）
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})

    messages.append(
        {
            "role": "user",
            "content": build_user_message_with_context(query, context),
        }
    )

    if stream:
        return _stream_chat_completion(client, messages)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def _stream_chat_completion(client: OpenAI, messages: List[dict]) -> str:
    """
    流式调用 Chat API 并实时打印到命令行。

    返回:
        拼接后的完整回复文本
    """
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        extra_body={"enable_thinking": False},
    )

    parts: List[str] = []
    print("\n助手: ", end="", flush=True)

    for chunk in stream:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None) or ""
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
    session: RAGSession,
    query: str,
    verbose: bool = True,
) -> str:
    """
    执行完整的一轮 RAG 对话。

    流程:
        1. text-embedding-v4 向量化用户问题
        2. FAISS 粗召回
        3. 重排序
        4. 组装上下文
        5. deepseek-v4-pro 生成回答
    """
    if verbose:
        print("\n[检索] 正在向量化问题并查询 FAISS 索引...")

    raw_results = faiss_search(client, query, top_n=RETRIEVE_TOP_N)

    if verbose:
        print(f"[检索] 粗召回 {len(raw_results)} 条候选")

    reranked = rerank_results(query, raw_results, top_k=RERANK_TOP_K)
    session.last_retrieved = reranked

    if verbose:
        print(f"[重排序] 选取 {len(reranked)} 条上下文片段:")
        for chunk in reranked:
            item = chunk.metadata
            location = format_location(item)
            loc_str = f" ({location})" if location else ""
            print(
                f"  · [{chunk.rank}] {item.get('display_name', '')}{loc_str}"
                f" | 综合分={chunk.score:.3f}"
            )

    context = build_context_block(reranked)

    if verbose:
        print(f"\n[生成] 调用 {CHAT_MODEL} 生成回答...")

    answer = generate_answer(
        client,
        query=query,
        context=context,
        history=session.history,
        stream=True,
    )

    # 更新对话历史（历史中只保留「纯问题」与「纯回答」，不含 RAG 上下文）
    session.history.append(ChatTurn(role="user", content=query))
    session.history.append(ChatTurn(role="assistant", content=answer))

    return answer


# =============================================================================
# 命令行交互
# =============================================================================


def print_welcome(config: dict) -> None:
    """打印启动欢迎信息与索引状态。"""
    print("=" * 64)
    print("  监管文件 RAG 助手 (Regulatory Assistant)")
    print("=" * 64)
    print(f"  向量模型: {EMBEDDING_MODEL}")
    print(f"  对话模型: {CHAT_MODEL}")
    print(f"  索引向量: {config.get('vector_count', '未知')} 条")
    if config.get("created_at"):
        print(f"  索引时间: {config['created_at']}")
    print(f"  粗召回/重排: Top {RETRIEVE_TOP_N} -> Top {RERANK_TOP_K}")
    print("-" * 64)
    print("  输入问题开始对话；特殊命令:")
    print("    /help   显示帮助")
    print("    /clear  清空对话历史")
    print("    /refs   显示上一轮检索引用")
    print("    quit    退出程序")
    print("=" * 64)


def print_help() -> None:
    """打印帮助信息。"""
    print(
        "\n可用命令:\n"
        "  /help   — 显示此帮助\n"
        "  /clear  — 清空多轮对话历史\n"
        "  /refs   — 查看上一轮回答所引用的监管文件片段\n"
        "  quit / exit / 退出 — 结束程序\n"
        "\n每次提问都会自动执行 RAG 检索，无需额外命令。"
    )


def print_last_references(session: RAGSession) -> None:
    """展示上一轮检索到的引用片段摘要。"""
    if not session.last_retrieved:
        print("\n暂无检索引用记录，请先提问。")
        return

    print("\n上一轮检索引用:")
    for chunk in session.last_retrieved:
        item = chunk.metadata
        location = format_location(item)
        print("\n" + "-" * 60)
        print(f"[{chunk.rank}] {item.get('display_name', '')}")
        if location:
            print(f"定位: {location}")
        print(f"向量分: {chunk.vector_score:.4f} | 关键词分: {chunk.keyword_score:.4f} | 综合分: {chunk.score:.4f}")
        preview = truncate_text(item.get("text", "").replace("\n", " "), 200)
        print(f"内容: {preview}")


def run_interactive(client: OpenAI, session: RAGSession) -> None:
    """
    交互式命令行对话主循环。

    用户可持续输入问题；支持 /help、/clear、/refs 等内置命令。
    """
    print("\n已进入对话模式，请输入您的问题。\n")

    while True:
        try:
            user_input = input("您 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in {"quit", "exit", "q", "退出"}:
            print("\n再见！")
            break

        if lower_input == "/help":
            print_help()
            continue

        if lower_input == "/clear":
            session.history.clear()
            session.last_retrieved.clear()
            print("\n[已清空对话历史与引用记录]")
            continue

        if lower_input == "/refs":
            print_last_references(session)
            continue

        try:
            run_rag_turn(client, session, user_input, verbose=True)
        except Exception as exc:  # noqa: BLE001 - 单次失败不终止交互
            print(f"\n[错误] {exc}")


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="监管文件 RAG 命令行助手 — FAISS 检索 + deepseek-v4-pro 对话"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="单次提问（省略则进入交互模式）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="单次模式下隐藏检索过程日志",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""
    args = parse_args()

    try:
        config = ensure_index_ready()
        api_key = load_api_key()
        client = create_client(api_key)
        session = RAGSession()

        if args.query:
            print_welcome(config)
            run_rag_turn(client, session, args.query.strip(), verbose=not args.quiet)
            return

        print_welcome(config)
        run_interactive(client, session)

    except Exception as exc:  # noqa: BLE001 - 统一输出错误信息
        print(f"\n[错误] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
