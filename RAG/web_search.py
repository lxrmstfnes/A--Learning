#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识库未命中时的互联网检索补充。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence
from urllib.parse import urlparse

WEB_SEARCH_MAX_RESULTS = 5
WEB_SNIPPET_MAX_CHARS = 600
WEB_CONTEXT_MAX_CHARS = 5000
WEB_REQUEST_TIMEOUT = 12

# 优先权威来源（域名子串匹配）
TRUSTED_DOMAIN_HINTS = (
    "gov.cn",
    "cbirc.gov.cn",
    "nfra.gov.cn",
    "pbc.gov.cn",
    "safe.gov.cn",
    "csrc.gov.cn",
    "npc.gov.cn",
    "chinalaw.gov.cn",
    "12371.cn",
)


@dataclass
class WebSearchHit:
    rank: int
    title: str
    url: str
    snippet: str
    domain: str = ""


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return ""


def _is_trusted(url: str) -> bool:
    domain = _domain(url)
    return any(hint in domain for hint in TRUSTED_DOMAIN_HINTS)


def search_web(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> List[WebSearchHit]:
    """使用 DuckDuckGo 检索互联网文本结果。"""
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise ImportError(
            "未安装 duckduckgo-search，请执行: pip install duckduckgo-search"
        ) from exc

    hits: List[WebSearchHit] = []
    try:
        with DDGS() as ddgs:
            raw = list(
                ddgs.text(
                    query,
                    region="cn-zh",
                    max_results=max(max_results * 2, 8),
                )
            )
    except Exception:  # noqa: BLE001
        return []

    # 权威来源优先，其余按原序
    raw.sort(key=lambda item: (0 if _is_trusted(item.get("href", "")) else 1))

    for item in raw:
        url = (item.get("href") or item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        if not url or not snippet:
            continue
        hits.append(
            WebSearchHit(
                rank=len(hits) + 1,
                title=title or url,
                url=url,
                snippet=snippet[:WEB_SNIPPET_MAX_CHARS],
                domain=_domain(url),
            )
        )
        if len(hits) >= max_results:
            break

    return hits


def serialize_web_hits(hits: Sequence[WebSearchHit]) -> List[dict]:
    return [
        {
            "rank": hit.rank,
            "title": hit.title,
            "url": hit.url,
            "domain": hit.domain,
            "snippet": hit.snippet,
        }
        for hit in hits
    ]


def build_web_context_block(hits: Sequence[WebSearchHit]) -> str:
    if not hits:
        return "（互联网检索未返回可用结果。）"

    blocks: List[str] = []
    total = 0
    for hit in hits:
        block = (
            f"[网络 {hit.rank}] 标题: {hit.title}\n"
            f"链接: {hit.url}\n"
            f"摘要: {hit.snippet}"
        )
        if total + len(block) > WEB_CONTEXT_MAX_CHARS:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)
