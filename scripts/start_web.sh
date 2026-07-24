#!/usr/bin/env bash
# 运维挑战赛 Web 常驻启动（gunicorn）
# 路径自动跟随本仓库位置，换目录克隆后无需改脚本。
#
# 用法（在任意位置执行均可）:
#   cd /root/A--Learning          # 或你的实际路径
#   chmod +x scripts/start_web.sh
#   export DASHSCOPE_API_KEY="你的key"
#   ./scripts/start_web.sh --port 443
#   ./scripts/start_web.sh --stop

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-443}"
WORKERS="${WORKERS:-2}"
TIMEOUT="${TIMEOUT:-180}"
PID_FILE="${PID_FILE:-$ROOT_DIR/logs/web_app.pid}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/web_app.log}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

mkdir -p "$ROOT_DIR/logs"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --stop)
      if [[ -f "$PID_FILE" ]]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "已停止 Web 服务（$ROOT_DIR）"
      else
        pkill -f "gunicorn .*web_app:app" 2>/dev/null || true
        echo "已尝试停止 gunicorn web_app"
      fi
      exit 0
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 加载环境变量（不依赖固定机器路径）
# ---------------------------------------------------------------------------
load_env_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  # 仅加载 KEY=VALUE，忽略注释与空行
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[[:space:]]*export[[:space:]]+ ]]; then
      line="${line#*export }"
    fi
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      # shellcheck disable=SC2163
      export "$line"
    fi
  done < "$file"
}

load_env_file "$ENV_FILE"
load_env_file "/etc/rag-web.env"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  for f in "$HOME/.zshenv" "$HOME/.bashrc" "$HOME/.profile"; do
    [[ -f "$f" ]] || continue
    # shellcheck disable=SC1090
    source "$f" >/dev/null 2>&1 || true
  done
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[错误] 未设置 DASHSCOPE_API_KEY"
  echo "  可任选其一："
  echo "  1) export DASHSCOPE_API_KEY=xxx"
  echo "  2) 在仓库根目录写 .env：  echo 'DASHSCOPE_API_KEY=xxx' > $ROOT_DIR/.env"
  echo "  3) 写系统文件：          echo 'DASHSCOPE_API_KEY=xxx' | sudo tee /etc/rag-web.env"
  exit 1
fi

# ---------------------------------------------------------------------------
# 定位 Python / gunicorn（避免 sudo 后 PATH 丢包）
# ---------------------------------------------------------------------------
resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo ""
}

PYTHON_BIN="$(resolve_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[错误] 找不到 python3/python"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import gunicorn" >/dev/null 2>&1; then
  echo "[提示] 当前 Python ($PYTHON_BIN) 未安装 gunicorn，正在安装…"
  "$PYTHON_BIN" -m pip install -U gunicorn || {
    echo "[错误] 安装失败，请手动执行: $PYTHON_BIN -m pip install gunicorn"
    exit 1
  }
fi

GUNICORN_CMD=("$PYTHON_BIN" -m gunicorn)

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "检测到已有进程 PID=$(cat "$PID_FILE")，先停止..."
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  sleep 1
fi

echo "=============================================="
echo "  项目目录: $ROOT_DIR"
echo "  Python:   $PYTHON_BIN"
echo "  访问地址: http://${HOST}:${PORT}"
echo "  日志:     $LOG_FILE"
echo "=============================================="

nohup "${GUNICORN_CMD[@]}" \
  -w "$WORKERS" \
  -b "${HOST}:${PORT}" \
  --chdir "$ROOT_DIR" \
  --timeout "$TIMEOUT" \
  --pid "$PID_FILE" \
  --access-logfile "$ROOT_DIR/logs/access.log" \
  --error-logfile "$LOG_FILE" \
  --capture-output \
  web_app:app \
  >>"$LOG_FILE" 2>&1 &

sleep 1
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "启动成功 PID=$(cat "$PID_FILE")"
  echo "查看日志: tail -f $LOG_FILE"
  echo "停止服务: $ROOT_DIR/scripts/start_web.sh --stop"
else
  echo "[错误] 启动失败，请查看日志: $LOG_FILE"
  tail -n 50 "$LOG_FILE" || true
  exit 1
fi
