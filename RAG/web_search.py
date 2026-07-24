#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识库未命中时的互联网检索补充（运维挑战赛优先权威/运维相关来源）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence
from urllib.parse import urlparse

WEB_SEARCH_MAX_RESULTS = 6
WEB_SNIPPET_MAX_CHARS = 600
WEB_CONTEXT_MAX_CHARS = 5000
WEB_REQUEST_TIMEOUT = 12

# 优先权威 / 运维相关来源（域名子串匹配）
TRUSTED_DOMAIN_HINTS = (
    # 监管与政务
    "gov.cn",
    "cbirc.gov.cn",
    "nfra.gov.cn",
    "pbc.gov.cn",
    "safe.gov.cn",
    "csrc.gov.cn",
    "npc.gov.cn",
    "chinalaw.gov.cn",
    # 标准与厂商文档
    "iso.org",
    "ietf.org",
    "microsoft.com",
    "learn.microsoft.com",
    "redhat.com",
    "oracle.com",
    "huawei.com",
    "aliyun.com",
    "tencent.com",
    "huaweicloud.com",
    # 运维/技术社区（次优先，但仍常用）
    "cnblogs.com",
    "juejin.cn",
    "csdn.net",
    "zhihu.com",
    "segmentfault.com",
)

OPS_TOPIC_HINTS = (
    "变更管理",
    "变更",
    "发布",
    "升级",
    "补丁",
    "机房",
    "数据中心",
    "运维",
    "值班",
    "应急",
    "故障",
    "巡检",
    "备份",
    "容灾",
    "监控",
    "权限",
    "账号",
    "基础软件",
    "操作系统",
    "中间件",
    "数据库",
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


def build_ops_search_query(query: str) -> str:
    """为运维挑战赛补充检索词，提高公开资料命中率。"""
    text = (query or "").strip()
    if not text:
        return text

    lower = text.lower()
    # 已含足够运维语境时不重复堆砌
    if any(hint in text for hint in OPS_TOPIC_HINTS):
        suffix = " 银行 科技 运维 规范"
    else:
        suffix = " 银行 信息科技 运维 变更管理"

    # 避免过长查询
    candidate = f"{text}{suffix}"
    if len(candidate) > 120:
        return text
    if suffix.strip() in lower:
        return text
    return candidate


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
                    max_results=max(max_results * 2, 10),
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
        # 过滤明显无关广告页
        if re.search(r"(广告|招聘|彩票)", title):
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
