#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Agent 工作流 (LangGraph)
============================

基于 LangGraph 的三步 RAG 工作流，逻辑与 main.py 一致：

    1. Query 改写  — 优化检索用词（多轮追问也会在此扩展）
    2. 知识库检索  — FAISS 向量检索 + 相邻片段扩展
    3. 生成结论    — 基于检索片段调用 deepseek-v4-pro 回答

用法:
    python deliberative_rag_agent.py
    python deliberative_rag_agent.py --mode llm --query "客户经理考核标准是什么？"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

RAG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RAG_ROOT))

import main as rag  # noqa: E402

WEB_FALLBACK = True


# =============================================================================
# 状态
# =============================================================================


class RAGAgentState(TypedDict):
    user_query: str
    mode: str
    history: List[Dict[str, str]]

    # 步骤 1：改写
    rewrite: Optional[Dict[str, Any]]
    retrieval_query: Optional[str]

    # 步骤 2：检索
    hits: Optional[List[Dict[str, Any]]]
    context_block: Optional[str]
    web_hits: Optional[List[Dict[str, Any]]]

    # 步骤 3：结论
    final_answer: Optional[str]
    used_web_fallback: bool

    current_step: Literal["rewrite", "retrieve", "generate", "done"]
    error: Optional[str]


# =============================================================================
# 三步节点
# =============================================================================


def rewrite_node(
    state: RAGAgentState,
    *,
    client: OpenAI,
    session: rag.RAGSession,
) -> RAGAgentState:
    """步骤 1：Query 改写。"""
    print("1. Query 改写...")

    try:
        rewrite = rag.rewrite_query_for_retrieval(client, session, state["user_query"])
        session.last_rewrite = rewrite

        if rewrite.retrieval_query != state["user_query"]:
            print(f"   改写后: {rewrite.retrieval_query}")
            if len(rewrite.sub_queries) > 1:
                print(f"   子查询: {rewrite.sub_queries}")
        else:
            print("   无需改写")

        return {
            **state,
            "rewrite": {
                "original_query": rewrite.original_query,
                "retrieval_query": rewrite.retrieval_query,
                "sub_queries": rewrite.sub_queries,
                "detected_type": rewrite.detected_type,
                "confidence": rewrite.confidence,
            },
            "retrieval_query": rewrite.retrieval_query,
            "current_step": "retrieve",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {**state, "error": f"Query 改写失败: {exc}", "current_step": "rewrite"}


def retrieve_node(
    state: RAGAgentState,
    *,
    client: OpenAI,
    index,
    metadata,
    session: rag.RAGSession,
) -> RAGAgentState:
    """步骤 2：搜索知识库。"""
    print("2. 搜索知识库...")

    if state.get("error"):
        return state

    try:
        rewrite_data = state.get("rewrite") or {}
        sub_queries = rewrite_data.get("sub_queries") or [state["user_query"]]

        # 重建 RewriteResult 供 resolve_retrieval_hits 使用
        rewrite = rag.RewriteResult(
            original_query=state["user_query"],
            retrieval_query=state.get("retrieval_query") or state["user_query"],
            sub_queries=sub_queries,
            detected_type=rewrite_data.get("detected_type", ""),
            confidence=float(rewrite_data.get("confidence", 0) or 0),
        )

        raw_hits = rag.search_index_multi(
            client, index, metadata, sub_queries, top_k=rag.RETRIEVE_TOP_K
        )
        hits, retrieval_query = rag.resolve_retrieval_hits(
            session, state["user_query"], raw_hits, rewrite
        )
        if session.source_chunk_index is None:
            session.source_chunk_index = rag.build_source_chunk_index(metadata)
        hits = rag.finalize_retrieval_hits(hits, metadata, session.source_chunk_index)
        session.last_hits = hits

        print(f"   命中 {len(hits)} 条")
        for hit in hits[:3]:
            name = hit.metadata.get("display_name", "未知")
            print(f"   #{hit.rank} {name} (相似度 {hit.score:.3f})")

        return {
            **state,
            "hits": rag.serialize_hits(hits),
            "context_block": rag.build_context_block(hits),
            "retrieval_query": retrieval_query,
            "current_step": "generate",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {**state, "error": f"知识库检索失败: {exc}", "current_step": "retrieve"}


def generate_node(
    state: RAGAgentState,
    *,
    client: OpenAI,
    session: rag.RAGSession,
) -> RAGAgentState:
    """步骤 3：生成结论。"""
    print("3. 生成结论...")

    if state.get("error"):
        return state

    try:
        query = state["user_query"]
        context = state.get("context_block") or ""
        retrieval_query = state.get("retrieval_query") or query
        history = [rag.ChatTurn(role=t["role"], content=t["content"]) for t in state.get("history", [])]

        # 从 session 恢复 RetrievedHit（供联网兜底判断）
        hits = session.last_hits
        web_hits: List[rag.WebSearchHit] = []
        used_web = False

        if WEB_FALLBACK and rag.should_prefetch_web(query, hits):
            answer, web_hits, used_web = rag.answer_with_web_if_needed(
                client, query, retrieval_query, hits, state["mode"], history
            )
            if not used_web:
                answer = rag.generate_answer_sync(client, query, context, state["mode"], history)
        else:
            local_answer = rag.generate_answer_sync(client, query, context, state["mode"], history)
            answer, web_hits, used_web = rag.answer_with_web_if_needed(
                client, query, retrieval_query, hits, state["mode"], history, local_answer=local_answer
            )
            if not used_web:
                answer = local_answer

        if used_web:
            print("   [联网] 已补充互联网检索")

        updated_history = list(state.get("history", []))
        updated_history.append({"role": "user", "content": query})
        updated_history.append({"role": "assistant", "content": answer})

        return {
            **state,
            "final_answer": answer,
            "web_hits": rag.serialize_web_hits(web_hits),
            "used_web_fallback": used_web,
            "history": updated_history,
            "current_step": "done",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {**state, "error": f"生成结论失败: {exc}", "current_step": "generate"}


# =============================================================================
# Agent 封装
# =============================================================================


class RAGAgent:
    """三步 RAG LangGraph Agent。"""

    def __init__(
        self,
        mode: str = "normal",
        *,
        client: Optional[OpenAI] = None,
        index=None,
        metadata: Optional[List[dict]] = None,
    ) -> None:
        self.mode = mode
        self.client = client or rag.create_client(rag.load_api_key())
        if index is None or metadata is None:
            index_dir = rag.MODE_CONFIG[mode]["index_dir"]
            self.index, self.metadata, self.config = rag.load_index_bundle(index_dir)
        else:
            self.index = index
            self.metadata = metadata
            self.config = {}
        self.session = rag.RAGSession(mode=mode)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGAgentState)

        workflow.add_node(
            "rewrite",
            lambda s: rewrite_node(s, client=self.client, session=self.session),
        )
        workflow.add_node(
            "retrieve",
            lambda s: retrieve_node(
                s, client=self.client, index=self.index, metadata=self.metadata, session=self.session
            ),
        )
        workflow.add_node(
            "generate",
            lambda s: generate_node(s, client=self.client, session=self.session),
        )

        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def invoke(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        initial: RAGAgentState = {
            "user_query": query,
            "mode": self.mode,
            "history": history or [],
            "rewrite": None,
            "retrieval_query": None,
            "hits": None,
            "context_block": None,
            "web_hits": None,
            "final_answer": None,
            "used_web_fallback": False,
            "current_step": "rewrite",
            "error": None,
        }
        return self.graph.invoke(initial)

    def print_mermaid(self) -> None:
        print(self.graph.get_graph().draw_mermaid())


# 兼容旧名称
DeliberativeRAGAgent = RAGAgent


# =============================================================================
# CLI
# =============================================================================


def run_interactive(agent: RAGAgent) -> None:
    print("\n已进入 RAG Agent 对话模式（输入 quit 退出，/graph 查看流程图）\n")
    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("您 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        lower = user_input.lower()
        if lower in {"quit", "exit", "q", "退出"}:
            print("再见！")
            break
        if lower == "/graph":
            agent.print_mermaid()
            continue

        print("\n--- RAG Agent 开始处理 ---")
        try:
            result = agent.invoke(user_input, history=history)
            history = result.get("history", history)
            if result.get("error"):
                print(f"\n[错误] {result['error']}")
            else:
                print(f"\n助手: {result.get('final_answer', '')}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[错误] {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Agent — LangGraph 三步工作流")
    parser.add_argument(
        "--mode",
        choices=["competition", "llm", "normal"],
        default="",
        help="知识库模式（默认优先 competition）",
    )
    parser.add_argument("--query", default="", help="单次提问")
    parser.add_argument("--top-k", type=int, default=rag.RETRIEVE_TOP_K, help="检索条数")
    parser.add_argument("--no-web", action="store_true", help="禁用联网补充")
    parser.add_argument("--graph", action="store_true", help="打印流程图后退出")
    return parser.parse_args()


def main() -> None:
    global WEB_FALLBACK  # noqa: PLW0603
    args = parse_args()
    rag.RETRIEVE_TOP_K = args.top_k
    WEB_FALLBACK = not args.no_web

    mode = args.mode or rag.choose_mode_interactive()
    index_dir = rag.MODE_CONFIG[mode]["index_dir"]

    try:
        index, metadata, config = rag.load_index_bundle(index_dir)
        client = rag.create_client(rag.load_api_key())
        agent = RAGAgent(mode=mode, client=client, index=index, metadata=metadata)

        print("=" * 64)
        print("  RAG Agent（LangGraph 三步工作流）")
        print("=" * 64)
        print(f"  模式: {rag.MODE_CONFIG[mode]['label']}")
        print(f"  索引: {index_dir}")
        print(f"  流程: Query改写 → 知识库检索 → 生成结论")
        print(f"  回答模型: {rag.CHAT_MODEL}")
        print(f"  联网补充: {'启用' if WEB_FALLBACK else '关闭'}")
        print("=" * 64)

        if args.graph:
            agent.print_mermaid()
            return

        if args.query:
            print("\n--- RAG Agent 开始处理 ---")
            result = agent.invoke(args.query.strip())
            if result.get("error"):
                print(f"\n[错误] {result['error']}")
                sys.exit(1)
            print(f"\n助手: {result.get('final_answer', '')}\n")
            return

        run_interactive(agent)

    except Exception as exc:  # noqa: BLE001
        print(f"\n[错误] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
