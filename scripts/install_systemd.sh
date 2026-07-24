#!/usr/bin/env bash
# 按「当前仓库真实路径」生成并安装 systemd 服务（目录换了也能用）
#
# 用法:
#   cd /root/A--Learning    # 任意实际路径
#   chmod +x scripts/install_systemd.sh
#   export DASHSCOPE_API_KEY="你的key"   # 或先写好 .env
#   sudo ./scripts/install_systemd.sh
#   sudo ./scripts/install_systemd.sh --port 443

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-443}"
WORKERS="${WORKERS:-2}"
TIMEOUT="${TIMEOUT:-180}"
SERVICE_NAME="${SERVICE_NAME:-rag-web}"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_SYSTEM="/etc/rag-web.env"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --name) SERVICE_NAME="$2"; UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[错误] 请使用 sudo 运行本脚本"
  exit 1
fi

# 定位 python（优先当前环境）
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[错误] 找不到 python3"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import gunicorn" >/dev/null 2>&1; then
  echo "[提示] 安装 gunicorn…"
  "$PYTHON_BIN" -m pip install -U gunicorn
fi

mkdir -p "$ROOT_DIR/logs"

# 写入系统环境变量（若仓库有 .env 则优先同步 DASHSCOPE）
if [[ ! -f "$ENV_SYSTEM" ]]; then
  if [[ -f "$ROOT_DIR/.env" ]]; then
    grep -E '^(DASHSCOPE_API_KEY|OPENAI_API_KEY)=' "$ROOT_DIR/.env" > "$ENV_SYSTEM" || true
  fi
fi
if [[ ! -f "$ENV_SYSTEM" || ! -s "$ENV_SYSTEM" ]]; then
  if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}" > "$ENV_SYSTEM"
  else
    echo "[警告] 未检测到 API Key。请稍后编辑: $ENV_SYSTEM"
    echo "DASHSCOPE_API_KEY=" > "$ENV_SYSTEM"
  fi
fi
chmod 600 "$ENV_SYSTEM"

cat > "$UNIT_PATH" <<EOF
[Unit]
Description=运维挑战赛 RAG Web 问答服务
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${ROOT_DIR}
EnvironmentFile=-${ENV_SYSTEM}
ExecStart=${PYTHON_BIN} -m gunicorn -w ${WORKERS} -b ${HOST}:${PORT} --chdir ${ROOT_DIR} --timeout ${TIMEOUT} --access-logfile ${ROOT_DIR}/logs/access.log --error-logfile ${ROOT_DIR}/logs/web_app.log web_app:app
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

echo "=============================================="
echo "  已安装 systemd 服务: $SERVICE_NAME"
echo "  项目目录: $ROOT_DIR"
echo "  单元文件: $UNIT_PATH"
echo "  环境变量: $ENV_SYSTEM"
echo "  访问:     http://${HOST}:${PORT}"
echo "=============================================="
echo "状态: systemctl status $SERVICE_NAME"
echo "日志: journalctl -u $SERVICE_NAME -f"
echo "重启: systemctl restart $SERVICE_NAME"
echo "停止: systemctl stop $SERVICE_NAME"
echo
echo "若以后把仓库挪到新目录，进入新目录再执行一次:"
echo "  sudo $ROOT_DIR/scripts/install_systemd.sh --port $PORT"
