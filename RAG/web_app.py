#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库 Web 问答界面 — 监管知识应知应会
======================

在项目根目录运行，提供浏览器问答界面，支持普通 / LLM 两种向量库模式。

启动:
    pip install -r requirements-web.txt
    export DASHSCOPE_API_KEY="your-key"
    python web_app.py

云服务器部署（443 端口，需 root 或 setcap）:

    # HTTPS（推荐，443 标准用法）
    sudo python3 web_app.py --host 0.0.0.0 --port 443 \\
        --ssl-cert /path/to/cert.pem --ssl-key /path/to/key.pem

    # 仅 HTTP（测试用，生产建议配 SSL 或前面加 Nginx）
    sudo python3 web_app.py --host 0.0.0.0 --port 443

本地调试可改用高位端口:
    python3 web_app.py --port 8080

前置条件:
    已构建向量库 — Normal/GetKnowledge.py 或 LLM/GetKnowledgeLLM.py
"""

from __future__ import annotations

import argparse
import secrets
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request, session

ROOT_DIR = Path(__file__).resolve().parent
RAG_DIR = ROOT_DIR / "RAG"
sys.path.insert(0, str(RAG_DIR))

import main as rag  # noqa: E402

# Web 端仅使用本地知识库，不启用联网检索
rag.WEB_FALLBACK_ENABLED = False

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
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址（云服务器默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=443,
        help="监听端口（云服务器安全组已开 443 时默认 443）",
    )
    parser.add_argument(
        "--ssl-cert",
        default="",
        help="HTTPS 证书文件路径（与 --ssl-key 同时指定则启用 SSL）",
    )
    parser.add_argument(
        "--ssl-key",
        default="",
        help="HTTPS 私钥文件路径",
    )
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser.parse_args()


def build_ssl_context(cert: str, key: str) -> Optional[tuple[str, str]]:
    """校验 SSL 证书路径并返回 Flask ssl_context 参数。"""
    if not cert and not key:
        return None
    if not cert or not key:
        raise ValueError("启用 HTTPS 须同时指定 --ssl-cert 与 --ssl-key")

    cert_path = Path(cert).expanduser().resolve()
    key_path = Path(key).expanduser().resolve()
    if not cert_path.is_file():
        raise FileNotFoundError(f"证书文件不存在: {cert_path}")
    if not key_path.is_file():
        raise FileNotFoundError(f"私钥文件不存在: {key_path}")
    return str(cert_path), str(key_path)


def main() -> None:
    args = parse_args()

    try:
        ssl_context = build_ssl_context(args.ssl_cert, args.ssl_key)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)

    scheme = "https" if ssl_context else "http"
    display_host = args.host if args.host != "0.0.0.0" else "你的服务器IP"

    print("=" * 60)
    print("  监管知识应知应会 — Web 问答")
    print("=" * 60)
    print(f"  访问地址: {scheme}://{display_host}:{args.port}")
    print(f"  监听: {args.host}:{args.port}  SSL={'开启' if ssl_context else '未开启'}")
    if args.port < 1024:
        print("  提示: 443 等特权端口需 sudo 运行，或: setcap cap_net_bind_service=+ep $(which python3)")
    if args.port == 443 and not ssl_context:
        print("  提示: 443 端口建议配合 --ssl-cert / --ssl-key 使用 HTTPS")
    print("  请确保已设置 DASHSCOPE_API_KEY 并已构建 faiss_index")
    print("=" * 60)

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        ssl_context=ssl_context,
    )


if __name__ == "__main__":
    main()
