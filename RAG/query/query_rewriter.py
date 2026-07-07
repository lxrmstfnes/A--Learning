#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query 改写模块 — 供 RAG 检索前调用，使用轻量 LLM 优化检索查询。"""

from __future__ import annotations

import json
import re
from typing import List, Sequence, Union

from openai import OpenAI

QUERY_REWRITE_MODEL = "qwen-turbo-latest"


def _call_llm(client: OpenAI, prompt: str, model: str = QUERY_REWRITE_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json_response(text: str) -> dict:
    """从 LLM 输出中解析 JSON（兼容 markdown 代码块）。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def format_conversation_history(history: Sequence) -> str:
    """将 ChatTurn 列表格式化为改写 prompt 可用的对话历史。"""
    lines: List[str] = []
    for turn in history:
        role = "用户" if turn.role == "user" else "助手"
        lines.append(f"{role}: {turn.content.strip()}")
    return "\n".join(lines)


class QueryRewriter:
    """基于 LLM 的 Query 改写器，用于检索前优化用户问题。"""

    def __init__(self, client: OpenAI, model: str = QUERY_REWRITE_MODEL):
        self.client = client
        self.model = model

    def rewrite_context_dependent_query(self, current_query: str, conversation_history: str) -> str:
        instruction = (
            "你是金融监管知识库的智能查询优化助手。"
            "请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。"
            "如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。"
            "如果不依赖，直接返回原问题。只输出改写后的问题，不要解释。"
        )
        prompt = (
            f"### 指令 ###\n{instruction}\n\n"
            f"### 对话历史 ###\n{conversation_history}\n\n"
            f"### 当前问题 ###\n{current_query}\n\n"
            f"### 改写后的问题 ###\n"
        )
        return _call_llm(self.client, prompt, self.model)

    def rewrite_comparative_query(self, query: str, context_info: str) -> str:
        instruction = (
            "你是金融监管知识库的查询分析专家。"
            "请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象，"
            "然后将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。"
            "只输出改写后的查询，不要解释。"
        )
        prompt = (
            f"### 指令 ###\n{instruction}\n\n"
            f"### 对话历史/上下文信息 ###\n{context_info}\n\n"
            f"### 原始问题 ###\n{query}\n\n"
            f"### 改写后的查询 ###\n"
        )
        return _call_llm(self.client, prompt, self.model)

    def rewrite_ambiguous_reference_query(self, current_query: str, conversation_history: str) -> str:
        instruction = (
            "你是消除语言歧义的专家。"
            "请分析用户的当前问题和对话历史，找出问题中「都」「它」「这个」等模糊指代词具体指向的对象，"
            "然后将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。"
            "只输出改写后的问题，不要解释。"
        )
        prompt = (
            f"### 指令 ###\n{instruction}\n\n"
            f"### 对话历史 ###\n{conversation_history}\n\n"
            f"### 当前问题 ###\n{current_query}\n\n"
            f"### 改写后的问题 ###\n"
        )
        return _call_llm(self.client, prompt, self.model)

    def rewrite_multi_intent_query(self, query: str) -> List[str]:
        instruction = (
            "你是任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。"
            "以 JSON 数组格式输出，例如：[\"问题1\", \"问题2\"]。只输出 JSON 数组，不要解释。"
        )
        prompt = (
            f"### 指令 ###\n{instruction}\n\n"
            f"### 原始问题 ###\n{query}\n\n"
            f"### 分解后的问题列表 ###\n"
        )
        response = _call_llm(self.client, prompt, self.model)
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass
        return [response or query]

    def rewrite_rhetorical_query(self, current_query: str, conversation_history: str) -> str:
        instruction = (
            "你是沟通理解专家。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题，"
            "然后将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。"
            "只输出改写后的问题，不要解释。"
        )
        prompt = (
            f"### 指令 ###\n{instruction}\n\n"
            f"### 对话历史 ###\n{conversation_history}\n\n"
            f"### 当前问题 ###\n{current_query}\n\n"
            f"### 改写后的问题 ###\n"
        )
        return _call_llm(self.client, prompt, self.model)

    def auto_rewrite_query(
        self,
        query: str,
        conversation_history: str = "",
        context_info: str = "",
    ) -> dict:
        instruction = """
你是金融监管知识库的智能查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含「还有」「其他」「那」等需要上下文理解的词汇
2. 对比型 - 包含「哪个」「比较」「更」等比较词汇
3. 模糊指代型 - 包含「它」「他们」「都」「这个」等指代词
4. 多意图型 - 包含多个独立问题，用「、」或「？」分隔
5. 反问型 - 包含「不会」「难道」等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回 JSON 格式：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询（多意图型可为字符串或数组）",
    "confidence": 0.9
}
"""
        prompt = (
            f"### 指令 ###\n{instruction}\n\n"
            f"### 对话历史 ###\n{conversation_history}\n\n"
            f"### 上下文信息 ###\n{context_info}\n\n"
            f"### 原始查询 ###\n{query}\n\n"
            f"### 分析结果 ###\n"
        )
        response = _call_llm(self.client, prompt, self.model)
        result = _parse_json_response(response)
        if not result:
            return {"query_type": "未知类型", "rewritten_query": query, "confidence": 0.5}
        return result

    def auto_rewrite_and_execute(
        self,
        query: str,
        conversation_history: str = "",
        context_info: str = "",
    ) -> dict:
        """自动识别 Query 类型并执行对应改写策略。"""
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        query_type = str(result.get("query_type", ""))

        if "上下文依赖" in query_type:
            final_result = self.rewrite_context_dependent_query(query, conversation_history)
        elif "对比" in query_type:
            final_result = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif "模糊指代" in query_type:
            final_result = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif "多意图" in query_type:
            final_result = self.rewrite_multi_intent_query(query)
        elif "反问" in query_type:
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            final_result = result.get("rewritten_query", query)

        return {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get("confidence", 0.5),
            "rewritten_query": final_result,
        }


def normalize_rewritten_query(rewritten: Union[str, List[str], None], fallback: str) -> tuple[str, List[str]]:
    """
    将改写结果规范化为 (主检索查询, 子查询列表)。
    多意图时 sub_queries 含多条，否则仅含一条。
    """
    if rewritten is None:
        return fallback, [fallback]
    if isinstance(rewritten, list):
        sub_queries = [str(q).strip() for q in rewritten if str(q).strip()]
        if not sub_queries:
            return fallback, [fallback]
        return sub_queries[0], sub_queries
    text = str(rewritten).strip()
    if not text or text == fallback:
        return fallback, [fallback]
    return text, [text]
