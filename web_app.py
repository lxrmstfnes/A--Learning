#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库 Web 问答界面
======================

在项目根目录运行，提供浏览器问答界面，支持普通 / LLM 两种向量库模式。

启动:
    pip install -r requirements-web.txt
    export DASHSCOPE_API_KEY="your-key"
    python web_app.py

云服务器部署示例:
    python web_app.py --host 0.0.0.0 --port 8080

前置条件:
    已构建向量库 — Normal/GetKnowledge.py 或 LLM/GetKnowledgeLLM.py
"""

from __future__ import annotations

import argparse
import secrets
import sys
import traceback
from pathlib import Path
from typing import Dict, Tuple

from flask import Flask, jsonify, render_template, request, session

ROOT_DIR = Path(__file__).resolve().parent
RAG_DIR = ROOT_DIR / "RAG"
sys.path.insert(0, str(RAG_DIR))

import main as rag  # noqa: E402

app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))
app.secret_key = secrets.token_hex(32)

# session_id -> (RAGSession, client, index, metadata, config)
_chat_store: Dict[str, Tuple[rag.RAGSession, object, object, list, dict]] = {}


def _get_or_create_chat(session_id: str, mode: str):
    """获取或初始化指定模式的聊天会话。"""
    if session_id in _chat_store:
        stored = _chat_store[session_id]
        if stored[0].mode == mode:
            return stored

    index_dir = rag.MODE_CONFIG[mode]["index_dir"]
    index, metadata, config = rag.load_index_bundle(index_dir)
    api_key = rag.load_api_key()
    client = rag.create_client(api_key)
    rag_session = rag.RAGSession(mode=mode)
    bundle = (rag_session, client, index, metadata, config)
    _chat_store[session_id] = bundle
    return bundle


def _ensure_browser_session() -> str:
    if "chat_id" not in session:
        session["chat_id"] = secrets.token_hex(16)
    return session["chat_id"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/modes", methods=["GET"])
def api_modes():
    modes = []
    for key, cfg in rag.MODE_CONFIG.items():
        index_dir = cfg["index_dir"]
        ready = (index_dir / "knowledge.index").exists() and (index_dir / "metadata.pkl").exists()
        config = {}
        config_file = index_dir / "config.json"
        if config_file.exists():
            import json

            config = json.loads(config_file.read_text(encoding="utf-8"))
        modes.append(
            {
                "id": key,
                "label": cfg["label"],
                "ready": ready,
                "vector_count": config.get("vector_count"),
                "created_at": config.get("created_at"),
            }
        )
    return jsonify({"modes": modes})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    mode = str(data.get("mode", "normal")).strip()

    if not query:
        return jsonify({"error": "问题不能为空"}), 400
    if mode not in rag.MODE_CONFIG:
        return jsonify({"error": f"无效模式: {mode}"}), 400

    chat_id = _ensure_browser_session()

    try:
        rag_session, client, index, metadata, config = _get_or_create_chat(chat_id, mode)
        result = rag.rag_query(client, index, metadata, rag_session, query)
        result["vector_count"] = config.get("vector_count")
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/clear", methods=["POST"])
def api_clear():
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "normal")).strip()
    chat_id = _ensure_browser_session()

    if chat_id in _chat_store and _chat_store[chat_id][0].mode == mode:
        _chat_store[chat_id][0].history.clear()
        _chat_store[chat_id][0].last_hits.clear()

    return jsonify({"ok": True})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 知识库 Web 问答")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（云服务器用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("  RAG 知识库 Web 问答")
    print("=" * 60)
    print(f"  访问地址: http://{args.host}:{args.port}")
    print("  请确保已设置 DASHSCOPE_API_KEY 并已构建 faiss_index")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
