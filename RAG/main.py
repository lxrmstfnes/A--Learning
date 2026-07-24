#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库问答入口 (main)
=========================

在 RAG 根目录运行，支持选择知识库模式进行问答:
    - normal:      普通方法（规则切分）→ faiss_index/
    - llm:         LLM 方法（原知识库）→ faiss_index/llm/
    - competition: LLM 方法（competition）→ faiss_index/llm/competition/

每次回答前会先调用轻量模型改写 Query，再检索向量库，最后调用 deepseek-v4-pro 生成回答。

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
from query.query_rewriter import (  # noqa: E402
    QUERY_REWRITE_MODEL,
    QueryRewriter,
    format_conversation_history,
    normalize_rewritten_query,
)
from web_search import (  # noqa: E402
    WebSearchHit,
    build_web_context_block,
    search_web,
    serialize_web_hits,
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
        "label": "LLM 方法（原知识库）",
        "index_dir": DEFAULT_LLM_INDEX_DIR,
    },
    "competition": {
        "label": "LLM 方法（competition）",
        "index_dir": DEFAULT_LLM_INDEX_DIR / "competition",
    },
}

CHAT_MODEL = "deepseek-v4-pro"
QUERY_REWRITE_ENABLED = True
RETRIEVE_TOP_K = 5
MIN_SCORE = 0.50
# 低于 MIN_SCORE 时，若最高分仍高于此值才兜底返回 Top1，否则视为未命中
FALLBACK_SCORE_FLOOR = 0.42
MAX_CHUNK_PREVIEW = 120
MAX_CONTEXT_CHARS = 10000
MAX_CHUNK_CHARS = 2500
MAX_HISTORY_TURNS = 6
# 短追问长度上限；以「那/还有」等开头的较长追问也会识别
FOLLOW_UP_MAX_LEN = 80
NEIGHBOR_EXPAND_RADIUS = 1
MAX_EXPANDED_HITS = 10
# 本地知识库未覆盖时是否联网补充
WEB_FALLBACK_ENABLED = True


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
class RewriteResult:
    """Query 改写结果。"""

    original_query: str
    retrieval_query: str
    sub_queries: List[str]
    detected_type: str = ""
    confidence: float = 0.0


@dataclass
class RAGSession:
    mode: str
    history: List[ChatTurn] = field(default_factory=list)
    last_hits: List[RetrievedHit] = field(default_factory=list)
    source_chunk_index: Optional[Dict[str, List[Tuple[int, dict]]]] = None
    last_rewrite: Optional[RewriteResult] = None


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


def is_follow_up_query(query: str, history: Sequence[ChatTurn]) -> bool:
    """判断是否为依赖上下文的追问（如「就只有这三条吗」「那独立董事解聘呢」）。"""
    if not history:
        return False
    text = query.strip()
    long_follow_up_starts = (
        r"^那",
        r"^还有",
        r"^另外",
        r"^以及",
        r"^除此之外",
        r"^除此之外",
    )
    if any(re.search(p, text) for p in long_follow_up_starts):
        return True
    if len(text) > FOLLOW_UP_MAX_LEN:
        return False
    patterns = (
        r"^就(这|那|这些|那些)",
        r"^(还|只有|仅仅|是否|能不能|可不可以)",
        r"^(有没有|还有其他|别的呢|详细说说|展开说说|举例)",
        r"^(为什么|怎么|如何)",
        r".+[吗嘛呢？?]$",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def build_retrieval_query(session: RAGSession, query: str) -> str:
    """多轮对话时将短追问扩展为可独立检索的完整问题（规则兜底）。"""
    if not is_follow_up_query(query, session.history):
        return query

    prev_user = ""
    for turn in reversed(session.history):
        if turn.role == "user":
            prev_user = turn.content.strip()
            break

    if not prev_user:
        return query

    return f"{prev_user}。追问：{query}"


def rewrite_query_for_retrieval(
    client: OpenAI,
    session: RAGSession,
    query: str,
) -> RewriteResult:
    """调用轻量 LLM 改写 Query，供后续向量检索使用。"""
    if not QUERY_REWRITE_ENABLED:
        fallback = build_retrieval_query(session, query)
        return RewriteResult(
            original_query=query,
            retrieval_query=fallback,
            sub_queries=[fallback],
        )

    history_text = format_conversation_history(session.history)
    try:
        rewriter = QueryRewriter(client, model=QUERY_REWRITE_MODEL)
        result = rewriter.auto_rewrite_and_execute(
            query,
            conversation_history=history_text,
            context_info=history_text,
        )
        retrieval_query, sub_queries = normalize_rewritten_query(
            result.get("rewritten_query"), query
        )
        return RewriteResult(
            original_query=query,
            retrieval_query=retrieval_query,
            sub_queries=sub_queries,
            detected_type=str(result.get("detected_type", "")),
            confidence=float(result.get("confidence", 0.0) or 0.0),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[改写] LLM 改写失败，使用规则兜底: {exc}")
        fallback = build_retrieval_query(session, query)
        return RewriteResult(
            original_query=query,
            retrieval_query=fallback,
            sub_queries=[fallback],
            detected_type="规则兜底",
        )


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

    # 若全部被分数阈值过滤，仅当最高分仍高于兜底下限才保留 Top1
    if not hits and len(indices[0]) > 0 and indices[0][0] >= 0:
        best_score = float(scores[0][0])
        if best_score >= FALLBACK_SCORE_FLOOR:
            idx = int(indices[0][0])
            if idx < len(metadata):
                hits.append(
                    RetrievedHit(
                        rank=1,
                        vector_id=idx,
                        score=best_score,
                        metadata=metadata[idx],
                    )
                )

    return hits


def search_index_multi(
    client: OpenAI,
    index: faiss.Index,
    metadata: List[dict],
    sub_queries: Sequence[str],
    top_k: int = RETRIEVE_TOP_K,
    min_score: float = MIN_SCORE,
) -> List[RetrievedHit]:
    """对多条子查询分别检索，按 vector_id 去重并保留最高相似度。"""
    if len(sub_queries) <= 1:
        return search_index(
            client, index, metadata, sub_queries[0] if sub_queries else "", top_k, min_score
        )

    best_by_id: Dict[int, RetrievedHit] = {}
    for sub_q in sub_queries:
        for hit in search_index(client, index, metadata, sub_q, top_k, min_score):
            existing = best_by_id.get(hit.vector_id)
            if existing is None or hit.score > existing.score:
                best_by_id[hit.vector_id] = hit

    merged = sorted(best_by_id.values(), key=lambda item: item.score, reverse=True)[:top_k]
    for rank, hit in enumerate(merged, start=1):
        hit.rank = rank
    return merged


def _source_key(meta: dict) -> str:
    return meta.get("source_file") or meta.get("display_name") or meta.get("preprocess_file") or ""


def build_source_chunk_index(metadata: List[dict]) -> Dict[str, List[Tuple[int, dict]]]:
    """按源文件分组并排序，便于扩展相邻片段。"""
    index: Dict[str, List[Tuple[int, dict]]] = {}
    for vector_id, meta in enumerate(metadata):
        key = _source_key(meta)
        if not key:
            continue
        index.setdefault(key, []).append((vector_id, meta))
    for items in index.values():
        items.sort(key=lambda item: (item[1].get("char_start", item[0]), item[0]))
    return index


def expand_hits_with_neighbors(
    hits: List[RetrievedHit],
    metadata: List[dict],
    source_index: Dict[str, List[Tuple[int, dict]]],
    radius: int = NEIGHBOR_EXPAND_RADIUS,
    max_hits: int = MAX_EXPANDED_HITS,
) -> List[RetrievedHit]:
    """命中某片段后，自动并入同文件的前后相邻片段，避免条款/列表被切分漏答。"""
    if not hits:
        return hits

    seen: set[int] = set()
    expanded: List[RetrievedHit] = []

    def append(vector_id: int, score: float) -> None:
        if vector_id in seen or vector_id < 0 or vector_id >= len(metadata):
            return
        seen.add(vector_id)
        expanded.append(
            RetrievedHit(
                rank=len(expanded) + 1,
                vector_id=vector_id,
                score=score,
                metadata=metadata[vector_id],
            )
        )

    for hit in hits:
        append(hit.vector_id, hit.score)
        key = _source_key(hit.metadata)
        siblings = source_index.get(key)
        if not siblings:
            continue
        pos = next((i for i, (vid, _) in enumerate(siblings) if vid == hit.vector_id), None)
        if pos is None:
            continue
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(siblings):
                nvid, _ = siblings[neighbor_pos]
                append(nvid, hit.score * (0.98 - abs(offset) * 0.02))
        if len(expanded) >= max_hits:
            break

    expanded.sort(
        key=lambda item: (
            _source_key(item.metadata),
            item.metadata.get("char_start", item.vector_id),
        )
    )
    for rank, hit in enumerate(expanded, start=1):
        hit.rank = rank
    return expanded[:max_hits]


def finalize_retrieval_hits(
    hits: List[RetrievedHit],
    metadata: List[dict],
    source_index: Optional[Dict[str, List[Tuple[int, dict]]]] = None,
) -> List[RetrievedHit]:
    """检索后处理：扩展相邻片段。"""
    if not hits:
        return hits
    if source_index is None:
        source_index = build_source_chunk_index(metadata)
    return expand_hits_with_neighbors(hits, metadata, source_index)


def resolve_retrieval_hits(
    session: RAGSession,
    query: str,
    new_hits: List[RetrievedHit],
    rewrite: RewriteResult,
) -> Tuple[List[RetrievedHit], str]:
    """确定最终用于生成的检索结果，并返回实际用于向量化的查询文本。"""
    retrieval_query = rewrite.retrieval_query
    prev_hits = list(session.last_hits)

    if not is_follow_up_query(query, session.history):
        return new_hits, retrieval_query

    new_usable = new_hits and new_hits[0].score >= MIN_SCORE
    if new_usable:
        return new_hits, retrieval_query

    prev_usable = prev_hits and prev_hits[0].score >= MIN_SCORE
    if prev_usable:
        return prev_hits, retrieval_query

    return new_hits, retrieval_query


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
        print("  未找到相关向量（将拒答，不依据通用知识作答）。")
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
        return (
            "（本次未检索到与用户问题明显相关的文档片段。"
            "请如实告知用户目前知识库里没有查到对应内容，语气自然友好；"
            "可建议用户补充法规名称、条款号、业务场景等关键词后重问，"
            "或轻松邀请其换个相关话题试试；"
            "不要用通用知识或训练数据编造答案。）"
        )

    blocks: List[str] = []
    total_chars = 0

    # 同文件片段按原文顺序排列，便于模型读完整条款
    ordered_hits = sorted(
        hits,
        key=lambda hit: (
            _source_key(hit.metadata),
            hit.metadata.get("char_start", hit.vector_id),
        ),
    )

    for hit in ordered_hits:
        meta = hit.metadata
        display_name = meta.get("display_name", "未知文件")
        location = format_location(meta)
        raw_text = meta.get("text", "")
        body = raw_text if len(raw_text) <= MAX_CHUNK_CHARS else truncate_text(raw_text, MAX_CHUNK_CHARS)

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
        f"你是四川农商银行的一位严谨、亲切的知识库问答助手（当前知识库: {mode_label}）。\n"
        "你像一位靠谱的同事：答得准、说得明白，找不到时也愿意帮用户把问题问清楚。\n"
        "用户已在界面中看到检索到的文档片段，请遵守：\n"
        "【准确性 — 最高优先级】\n"
        "1. 依据参考片段中明确写出的内容作答；找到置信度高的对应文档时，不得编造、推测、"
        "补充或引申文档未提及的信息；\n"
        "2. 数字、金额、比例、日期、条款编号、文件名称等须与原文一致，不得改写或约化；\n"
        "3. 片段不足以支撑结论时，坦诚说明「目前知识库里没查到明确规定」或「现有材料还不足以确定」，"
        "不要自行猜测、调和或合并推断；\n"
        "4. 若提供了同一法规的多段连续片段，应合并理解并尽量完整作答（尤其列举型条款），"
        "不要因某段预览出现省略号就认为原文到此为止——以各片段完整正文为准；\n"
        "5. 重点回答用户所问，不跑题；拿不准就直接说暂时无法从文档确认。\n"
        "【找不到或信息不足时 — 语气要自然】\n"
        "6. 先简短说明没查到或信息不够，不要冷冰冰地拒答；\n"
        "7. 主动帮用户把问题问细一点，例如：请补充法规/文件全称、第几条、具体业务场景、"
        "涉及哪类机构或产品等；给 1～2 个具体提问示例即可，不要长篇大论；\n"
        "8. 可以自然收尾，如「您也可以换个相关话题试试，我帮您查查」——轻松、不敷衍；\n"
        "9. 即使没答案，也要让用户感到被接住，而不是被挡回去。\n"
        "【表达风格】\n"
        "10. 用自然、简洁的口语化书面语，像同事面对面解释；\n"
        "11. 不要以「根据提供的参考文档」「根据以上片段」等套话开头；\n"
        "12. 不要在回答末尾单独列出「来源」「章节」「页码」等引用块，不要标注【引用 N】；\n"
        "13. 能一句话说清就用一句话；需要分点时用简短条目；\n"
        "14. 使用简体中文。"
    )


def build_user_message(query: str, context: str) -> str:
    return (
        "以下是从知识库检索到的参考片段（这是你作答的主要依据；"
        "用户已看过检索结果，回答时不要重复标注来源）：\n"
        "-----\n"
        f"{context}\n"
        "-----\n\n"
        f"用户问题：{query}\n\n"
        "请基于以上片段用自然、亲切的中文直接回答。"
        "有依据就说清楚，列举型问题尽量列全片段中出现的条目；"
        "没有或不够时，如实说明并帮用户想想怎么问得更具体，"
        "语气像同事帮忙，不要猜测、不要编造。"
    )


INSUFFICIENT_ANSWER_PATTERNS = (
    "未找到",
    "无法确认",
    "无法确定",
    "无法从文档",
    "没有提到",
    "没有涉及",
    "不足以",
    "未检索到",
    "现有片段",
    "文档中未",
    "知识库中未",
    "没查到",
    "暂时无法",
    "暂时没",
    "信息不够",
    "还不足以",
)

ARTICLE_REF_PATTERN = re.compile(r"第[一二三四五六七八九十百零〇两\d]+条")


def hits_text_blob(hits: Sequence[RetrievedHit]) -> str:
    return "\n".join(hit.metadata.get("text", "") for hit in hits)


def query_article_refs_missing(query: str, hits: Sequence[RetrievedHit]) -> bool:
    """问题明确指向某条款，但检索片段中未出现该条款。"""
    refs = ARTICLE_REF_PATTERN.findall(query)
    if not refs:
        return False
    blob = hits_text_blob(hits)
    return any(ref not in blob for ref in refs)


def is_insufficient_local_answer(
    query: str,
    hits: Sequence[RetrievedHit],
    answer: str,
) -> bool:
    """判断本地知识库是否未能充分回答问题。"""
    if not hits or hits[0].score < MIN_SCORE:
        return True
    if query_article_refs_missing(query, hits):
        return True
    normalized = answer.strip()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in INSUFFICIENT_ANSWER_PATTERNS)


def build_web_system_prompt(mode: str) -> str:
    mode_label = MODE_CONFIG[mode]["label"]
    return (
        f"你是四川农商银行的一位严谨、亲切的知识库问答助手（本地知识库: {mode_label}）。\n"
        "本地知识库暂未覆盖用户所问，以下参考来自互联网公开检索，请结合摘要作答。\n"
        "须遵守：\n"
        "1. 优先依据检索摘要，不得编造法规条文；\n"
        "2. 开头自然说明「行里知识库暂时没查到，我帮您从公开资料里找了找」之类，不要生硬；\n"
        "3. 若网络摘要仍不够，坦诚说明，并建议用户补充法规名称、条款号等后再问；\n"
        "4. 条款、数字尽量与摘要一致；\n"
        "5. 结尾轻提醒以官方正式文件为准；\n"
        "6. 语气像同事帮忙，用自然简洁的简体中文。"
    )


def build_web_user_message(query: str, local_context: str, web_context: str) -> str:
    local_part = local_context if local_context else "（本地未检索到有效片段）"
    return (
        "【本地知识库片段】\n"
        f"{local_part}\n\n"
        "【互联网检索摘要】\n"
        f"{web_context}\n\n"
        f"用户问题：{query}\n\n"
        "请结合互联网检索摘要回答；本地片段若与问题无关可忽略。"
        "语气自然亲切，不要重复标注链接编号。"
    )


def generate_web_answer_sync(
    client: OpenAI,
    query: str,
    local_context: str,
    web_context: str,
    mode: str,
    history: Sequence[ChatTurn],
) -> str:
    messages: List[dict] = [{"role": "system", "content": build_web_system_prompt(mode)}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {"role": "user", "content": build_web_user_message(query, local_context, web_context)}
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def should_prefetch_web(query: str, hits: Sequence[RetrievedHit]) -> bool:
    """无需先调本地 LLM，可直接联网的情形。"""
    if not hits or hits[0].score < MIN_SCORE:
        return True
    return query_article_refs_missing(query, hits)


def maybe_web_fallback(
    client: OpenAI,
    query: str,
    retrieval_query: str,
    hits: Sequence[RetrievedHit],
    local_answer: str,
    mode: str,
    history: Sequence[ChatTurn],
) -> Tuple[str, List[WebSearchHit], bool]:
    """本地答案不足时联网检索并重新生成。返回 (最终答案, 网络命中, 是否使用了联网)。"""
    if not WEB_FALLBACK_ENABLED:
        return local_answer, [], False
    if should_prefetch_web(query, hits):
        return local_answer, [], False
    if not is_insufficient_local_answer(query, hits, local_answer):
        return local_answer, [], False

    return _answer_from_web(client, query, retrieval_query, hits, mode, history)


def answer_with_web_if_needed(
    client: OpenAI,
    query: str,
    retrieval_query: str,
    hits: Sequence[RetrievedHit],
    mode: str,
    history: Sequence[ChatTurn],
    local_answer: Optional[str] = None,
) -> Tuple[str, List[WebSearchHit], bool]:
    """统一入口：优先本地；不足则联网。"""
    if not WEB_FALLBACK_ENABLED:
        return local_answer or "", [], False

    if should_prefetch_web(query, hits):
        return _answer_from_web(client, query, retrieval_query, hits, mode, history)

    if local_answer is None:
        return "", [], False

    if is_insufficient_local_answer(query, hits, local_answer):
        return _answer_from_web(client, query, retrieval_query, hits, mode, history)

    return local_answer, [], False


def _answer_from_web(
    client: OpenAI,
    query: str,
    retrieval_query: str,
    hits: Sequence[RetrievedHit],
    mode: str,
    history: Sequence[ChatTurn],
) -> Tuple[str, List[WebSearchHit], bool]:
    search_query = retrieval_query or query
    web_hits = search_web(search_query)
    if not web_hits:
        return "", web_hits, False

    local_context = build_context_block(hits) if hits else ""
    web_context = build_web_context_block(web_hits)
    web_answer = generate_web_answer_sync(
        client, query, local_context, web_context, mode, history
    )
    return web_answer, web_hits, True


def serialize_hits(hits: Sequence[RetrievedHit]) -> List[dict]:
    """将检索结果序列化为 JSON 可传输结构。"""
    result: List[dict] = []
    for hit in hits:
        meta = hit.metadata
        result.append(
            {
                "rank": hit.rank,
                "vector_id": hit.vector_id,
                "chunk_id": meta.get("chunk_id"),
                "score": round(hit.score, 4),
                "display_name": meta.get("display_name", ""),
                "location": format_location(meta),
                "summary": meta.get("summary", ""),
                "preview": truncate_text(meta.get("text", ""), MAX_CHUNK_PREVIEW),
            }
        )
    return result


def generate_answer_sync(
    client: OpenAI,
    query: str,
    context: str,
    mode: str,
    history: Sequence[ChatTurn],
) -> str:
    """非流式生成回答（供 Web API 使用）。"""
    messages: List[dict] = [{"role": "system", "content": build_system_prompt(mode)}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": build_user_message(query, context)})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def rag_query(
    client: OpenAI,
    index: faiss.Index,
    metadata: List[dict],
    session: RAGSession,
    query: str,
    top_k: int = RETRIEVE_TOP_K,
) -> dict:
    """执行一轮 RAG 并返回结构化结果（供 Web / API 调用）。"""
    rewrite = rewrite_query_for_retrieval(client, session, query)
    session.last_rewrite = rewrite
    raw_hits = search_index_multi(
        client, index, metadata, rewrite.sub_queries, top_k=top_k
    )
    hits, retrieval_query = resolve_retrieval_hits(session, query, raw_hits, rewrite)
    if session.source_chunk_index is None:
        session.source_chunk_index = build_source_chunk_index(metadata)
    hits = finalize_retrieval_hits(hits, metadata, session.source_chunk_index)
    session.last_hits = hits
    context = build_context_block(hits)

    if should_prefetch_web(query, hits):
        answer, web_hits, used_web = answer_with_web_if_needed(
            client, query, retrieval_query, hits, session.mode, session.history
        )
        if not used_web:
            answer = generate_answer_sync(client, query, context, session.mode, session.history)
            web_hits = []
    else:
        local_answer = generate_answer_sync(client, query, context, session.mode, session.history)
        answer, web_hits, used_web = answer_with_web_if_needed(
            client,
            query,
            retrieval_query,
            hits,
            session.mode,
            session.history,
            local_answer=local_answer,
        )
        if not used_web:
            answer = local_answer

    session.history.append(ChatTurn(role="user", content=query))
    session.history.append(ChatTurn(role="assistant", content=answer))

    return {
        "mode": session.mode,
        "mode_label": MODE_CONFIG[session.mode]["label"],
        "query": query,
        "retrieval_query": retrieval_query if retrieval_query != query else None,
        "rewrite_type": rewrite.detected_type or None,
        "rewrite_confidence": rewrite.confidence or None,
        "sub_queries": rewrite.sub_queries if len(rewrite.sub_queries) > 1 else None,
        "hits": serialize_hits(hits),
        "web_hits": serialize_web_hits(web_hits),
        "used_web_fallback": used_web,
        "answer": answer,
    }


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
    """执行一轮 RAG：Query 改写 → 检索 → 展示向量 → 生成回答。"""
    print(f"\n[改写] 调用 {QUERY_REWRITE_MODEL} 优化检索查询...")
    rewrite = rewrite_query_for_retrieval(client, session, query)
    session.last_rewrite = rewrite
    if rewrite.retrieval_query != query:
        print(f"[改写] 类型: {rewrite.detected_type or '—'}  置信度: {rewrite.confidence:.2f}")
        print(f"[改写] 检索查询: {rewrite.retrieval_query}")
        if len(rewrite.sub_queries) > 1:
            print(f"[改写] 多意图子查询: {rewrite.sub_queries}")
    else:
        print("[改写] 无需改写，使用原问题检索")

    print(f"\n[检索] 正在查询 {MODE_CONFIG[session.mode]['label']} 向量库...")
    raw_hits = search_index_multi(
        client, index, metadata, rewrite.sub_queries, top_k=RETRIEVE_TOP_K
    )
    hits, retrieval_query = resolve_retrieval_hits(session, query, raw_hits, rewrite)
    if session.source_chunk_index is None:
        session.source_chunk_index = build_source_chunk_index(metadata)
    hits = finalize_retrieval_hits(hits, metadata, session.source_chunk_index)
    session.last_hits = hits

    print_retrieved_hits(hits, session.mode)

    context = build_context_block(hits)

    if should_prefetch_web(query, hits):
        print(f"[联网] 本地片段未覆盖问题，正在检索互联网...")
        answer, web_hits, used_web = answer_with_web_if_needed(
            client, query, retrieval_query, hits, session.mode, session.history
        )
        if used_web:
            for hit in web_hits:
                print(f"  - {hit.title} ({hit.url})")
            print(f"\n助手: {answer}\n")
        else:
            print(f"[生成] 联网未命中，调用 {CHAT_MODEL} 基于本地片段回答...\n")
            answer = generate_answer(client, query, context, session.mode, session.history)
            web_hits = []
    else:
        print(f"[生成] 调用 {CHAT_MODEL} 生成回答...\n")
        local_answer = generate_answer(client, query, context, session.mode, session.history)
        answer, web_hits, used_web = answer_with_web_if_needed(
            client,
            query,
            retrieval_query,
            hits,
            session.mode,
            session.history,
            local_answer=local_answer,
        )
        if used_web:
            print("\n[联网] 本地知识库未覆盖，已检索互联网补充：")
            for hit in web_hits:
                print(f"  - {hit.title} ({hit.url})")
            print(f"\n助手(联网): {answer}\n")
        else:
            answer = local_answer
            web_hits = []

    session.history.append(ChatTurn(role="user", content=query))
    session.history.append(ChatTurn(role="assistant", content=answer))
    return answer


# =============================================================================
# 模式选择与交互
# =============================================================================


def choose_mode_interactive() -> str:
    """交互式选择 RAG 模式。"""
    print("\n请选择 RAG 模式:")
    print("  1. 普通方法（规则切分）       → faiss_index/")
    print("  2. LLM 方法（原知识库）       → faiss_index/llm/")
    print("  3. LLM 方法（competition）    → faiss_index/llm/competition/")
    print("  q. 退出")

    mapping = {
        "1": "normal",
        "2": "llm",
        "3": "competition",
        "normal": "normal",
        "llm": "llm",
        "competition": "competition",
    }

    while True:
        try:
            choice = input("\n请输入选项 [1/2/3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            sys.exit(0)

        if choice in {"q", "quit", "exit", "退出"}:
            print("再见！")
            sys.exit(0)
        if choice in mapping:
            return mapping[choice]
        print("无效选项，请输入 1、2 或 3。")


def print_welcome(mode: str, config: dict) -> None:
    index_dir = MODE_CONFIG[mode]["index_dir"]
    print("=" * 64)
    print("  RAG 知识库问答")
    print("=" * 64)
    print(f"  模式: {MODE_CONFIG[mode]['label']}")
    print(f"  索引目录: {index_dir}")
    print(f"  向量模型: {config.get('embedding_model', EMBEDDING_MODEL)}")
    print(f"  对话模型: {CHAT_MODEL}")
    print(f"  改写模型: {QUERY_REWRITE_MODEL}" + ("（已启用）" if QUERY_REWRITE_ENABLED else "（已关闭）"))
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
    parser = argparse.ArgumentParser(description="RAG 知识库问答 — 支持普通 / LLM / competition")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["normal", "llm", "competition"],
        default="",
        help="知识库模式: normal / llm / competition（省略则交互选择）",
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
