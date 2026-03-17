#!/usr/bin/env bash
# ============================================================================
# start_aiidalab_qe.sh
# 目的: 一键在远程无界面 Ubuntu 服务器上启用 (可选) 代理 + 激活指定 conda 环境 + 启动 AiiDAlab (aiidalab-qe) Web UI
#       并输出本地浏览器访问 / SSH 端口转发指引, 支持后台守护 (tmux/nohup) 与自动选端口。
# ----------------------------------------------------------------------------
# 依赖: bash, (ana)conda 或 mamba, 已安装 aiidalab-qe 及其依赖, 可选 proxy_on.sh.
# 用法示例:
#   1) 前台启动:    ./start_aiidalab_qe.sh
#   2) 指定端口:    ./start_aiidalab_qe.sh -p 8890
#   3) 指定环境:    ./start_aiidalab_qe.sh -e quantum-espresso
#   4) 自动选择空闲端口 + 后台 tmux 会话: ./start_aiidalab_qe.sh -a -t
#   5) 开启代理(若同目录有 proxy_on.sh 并想自动执行): ./start_aiidalab_qe.sh -P
#   6) 无 tmux 仅 nohup 后台: ./start_aiidalab_qe.sh -n
#   7) 仅探测并打印可访问 URL 不启动: ./start_aiidalab_qe.sh --dry-run
# ----------------------------------------------------------------------------
# 生成访问 URL 方式:
#   默认绑定 0.0.0.0:<PORT> (可改 --bind)。
#   本地访问方式(建议): SSH 端口转发: ssh -N -L <LOCAL_PORT>:localhost:<PORT> user@remote
#   然后浏览器访问: http://localhost:<LOCAL_PORT>/lab?token=<TOKEN>
# ----------------------------------------------------------------------------
# 选项简介:
#   -e|--env ENV           指定 conda 环境名 (默认: quantum-espresso)
#   -p|--port PORT         指定端口 (默认: 8888; 若被占用自动 +1 或用 -a 自动择空)
#   -a|--auto-port         自动从起始端口向上找第一个空闲
#   -b|--bind HOST         绑定地址 (默认: 0.0.0.0) 可改 127.0.0.1 (配合 SSH 转发)
#   -t|--tmux              在 tmux 会话内后台运行 (会话名: aiidalab_qe)
#   -n|--nohup             使用 nohup 后台输出到 logs/aiidalab_qe_<PORT>.log
#   -P|--proxy             若同目录有 proxy_on.sh 则先执行它
#   -r|--reuse             若已有运行中的同端口 jupyter 不重复启动, 直接显示访问信息
#   -F|--force-restart     若已有 tmux 会话/进程仍保留, 强制终止并按当前参数重启
#   -Q|--bootstrap-qe      若缺失 aiidalab-qe 应用则自动 git clone
#   --lab                  强制 jupyter lab (默认优先 lab, 否则 notebook)
#   --notebook             使用经典 notebook
#   --dry-run              仅解析参数与探测端口, 不启动
#   -h|--help              查看帮助
# ----------------------------------------------------------------------------
# 可扩展: 若有 aiidalab CLI (aiidalab start/aiidalab launch) 可替换 JUPYTER_CMD 逻辑。
# ============================================================================
set -e -o pipefail  # 去掉全局 -u 以避免第三方 activate 脚本未定义变量报错

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# 默认参数
CONDA_ENV="quantum-espresso"
PORT=8888
AUTO_PORT=0
BIND_HOST="0.0.0.0"
USE_TMUX=0
USE_NOHUP=0
RUN_PROXY=0
REUSE=0
FORCE_RESTART=0
FORCE_LAB=0
FORCE_NOTEBOOK=0
DRY_RUN=0
SESSION_NAME="aiidalab_qe"
LOG_DIR="${SCRIPT_DIR}/logs"
DETECTED_HOST=""
APPS_DIR="/project/apps"
BOOTSTRAP_QE=0
PORT_FILE="${LOG_DIR}/aiidalab_qe.port"

usage() {
  sed -n '1,140p' "$0" | grep -E '^#' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env) CONDA_ENV="$2"; shift 2;;
    -p|--port) PORT="$2"; shift 2;;
    -a|--auto-port) AUTO_PORT=1; shift;;
    -b|--bind) BIND_HOST="$2"; shift 2;;
    -t|--tmux) USE_TMUX=1; shift;;
    -n|--nohup) USE_NOHUP=1; shift;;
    -P|--proxy) RUN_PROXY=1; shift;;
    -r|--reuse) REUSE=1; shift;;
    -F|--force-restart) FORCE_RESTART=1; shift;;
    -Q|--bootstrap-qe) BOOTSTRAP_QE=1; shift;;
    --lab) FORCE_LAB=1; shift;;
    --notebook) FORCE_NOTEBOOK=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 1;;
  esac
done

if [[ $FORCE_LAB -eq 1 && $FORCE_NOTEBOOK -eq 1 ]]; then
  echo "[WARN] 同时指定 --lab 与 --notebook, 以 --lab 为准" >&2
  FORCE_NOTEBOOK=0
fi

# ---------- 函数: 检测命令是否存在 ----------
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] 需要命令: $1" >&2; exit 2; }; }

# ---------- 函数: 探测本机主要 IPv4 ----------
detect_host_ip() {
  local ip=""
  if command -v ip >/dev/null 2>&1; then
    ip=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++){ if($i=="src"){print $(i+1); exit}}}')
  fi
  if [[ -z "$ip" ]]; then
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
  fi
  if [[ -z "$ip" ]]; then
    ip=$(grep -m1 -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' /proc/net/fib_trie 2>/dev/null | head -n1)
  fi
  echo "$ip"
}

# ---------- 函数: 选取空闲端口 ----------
find_free_port() {
  local p=$1
  while ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$p$"; do
    p=$((p+1))
  done
  echo $p
}

# ---------- 函数: 激活 conda 环境 ----------
activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未检测到 conda, 请先安装/初始化" >&2; exit 4; fi
  # 记录 nounset 状态 (若用户手动加了 set -u)
  local had_nounset=0
  if set -o | grep -q 'nounset *on'; then had_nounset=1; set +u; fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  if ! conda activate "$CONDA_ENV" 2>/dev/null; then
    echo "[WARN] 未找到环境 $CONDA_ENV, 列出可用:" >&2
    conda env list >&2 || true
    [[ $had_nounset -eq 1 ]] && set -u
    exit 3
  fi
  [[ $had_nounset -eq 1 ]] && set -u
  # 防止 Qt activate 脚本引用未设变量导致后续报错
  export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
  echo "[INFO] 已激活 conda 环境: $CONDA_ENV (QT_XCB_GL_INTEGRATION=$QT_XCB_GL_INTEGRATION)" >&2
}

# ---------- 函数: 生成 JUPYTER 启动命令 ----------
resolve_jupyter_cmd() {
  local allow_root=""
  if [[ $EUID -eq 0 ]]; then
    allow_root="--allow-root"
  fi
  local base="jupyter"
  if [[ $FORCE_NOTEBOOK -eq 1 ]]; then
    echo "$base notebook $allow_root --NotebookApp.token= --NotebookApp.password= --no-browser --ip=$BIND_HOST --port=$PORT"
    return
  fi
  if [[ $FORCE_LAB -eq 1 ]]; then
    echo "$base lab $allow_root --NotebookApp.token= --NotebookApp.password= --no-browser --ip=$BIND_HOST --port=$PORT"
    return
  fi
  # 优先 lab
  if $base lab --version >/dev/null 2>&1; then
    echo "$base lab $allow_root --NotebookApp.token= --NotebookApp.password= --no-browser --ip=$BIND_HOST --port=$PORT"
  else
    echo "$base notebook $allow_root --NotebookApp.token= --NotebookApp.password= --no-browser --ip=$BIND_HOST --port=$PORT"
  fi
}

# ---------- 函数: 检测是否已有运行 ----------
existing_jupyter() {
  pgrep -af "jupyter.*--port=$PORT" || true
}

# ---------- 函数: 打印访问指引 ----------
print_access_info() {
  local token_line token
  # 尝试提取 token (如果未禁用 token)
  token_line=$(jupyter server list 2>/dev/null | grep ":$PORT/" || true)
  if [[ -n "$token_line" ]]; then
    token=$(echo "$token_line" | sed -n 's/.*token=\([a-z0-9]\+\).*/\1/p')
  fi
  # 若未手动设置, 自动探测主机 IP
  if [[ -z "$DETECTED_HOST" ]]; then
    DETECTED_HOST=$(detect_host_ip)
  fi
  echo "============================================================";
  echo "启动时间: $START_TIME";
  echo "AiiDAlab(QE) 访问方式:";
  echo "  1) SSH 端口转发 (推荐, 本地执行):";
  echo "     ssh -N -L ${PORT}:127.0.0.1:${PORT} <USER>@<REMOTE_HOST>";
  if [[ -n "$DETECTED_HOST" ]]; then
    echo "     示例(已探测服务端 IP): ssh -N -L ${PORT}:127.0.0.1:${PORT} ${USER}@${DETECTED_HOST}";
  fi
  echo "  2) 浏览器访问: http://localhost:${PORT}/ (若 token 关闭)";
  if [[ -n "$token" ]]; then
    echo "     或: http://localhost:${PORT}/lab?token=${token}";
  fi
  if [[ "$BIND_HOST" == "0.0.0.0" && -n "$DETECTED_HOST" ]]; then
    echo "  3) 直接访问(当前绑定 0.0.0.0): http://${DETECTED_HOST}:${PORT}/";
  else
    echo "  3) 若绑定 0.0.0.0 且安全组允许, 可直接: http://<REMOTE_HOST>:${PORT}/";
  fi
  echo "日志位置: $LOG_DIR (若使用 nohup), 或 tmux 会话: $SESSION_NAME";
  echo "============================================================";
}

# ---------- 主流程 ----------

# 端口选择
if [[ $AUTO_PORT -eq 1 ]]; then
  PORT=$(find_free_port "$PORT")
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY] 端口: $PORT 绑定: $BIND_HOST 环境: $CONDA_ENV"
  exit 0
fi

# 可选代理
if [[ $RUN_PROXY -eq 1 ]]; then
  if [[ -x "${SCRIPT_DIR}/proxy_on.sh" ]]; then
    echo "[INFO] 执行 proxy_on.sh" >&2
    ( cd "$SCRIPT_DIR" && ./proxy_on.sh ) || echo "[WARN] 代理脚本执行失败" >&2
  else
    echo "[WARN] 未找到 proxy_on.sh, 跳过代理" >&2
  fi
fi

activate_conda_env

# ---------- 可选: 自动引导安装 aiidalab-qe 应用 ----------
bootstrap_qe_app() {
  # 若用户已有自定义 AIIDALAB_APPS 则使用之, 否则使用默认 /project/apps (存在) 或回退到脚本目录下 apps
  if [[ -z "${AIIDALAB_APPS:-}" ]]; then
    if [[ ! -d "$APPS_DIR" ]]; then
      # 若 /project 不存在则改用脚本目录下
      if [[ ! -d "/project" ]]; then
        APPS_DIR="${SCRIPT_DIR}/apps"
      fi
      mkdir -p "$APPS_DIR"
    fi
    export AIIDALAB_APPS="$APPS_DIR"
  else
    APPS_DIR="$AIIDALAB_APPS"
  fi
  if [[ ! -d "$APPS_DIR/aiidalab-qe" ]]; then
    echo "[INFO] 未检测到 aiidalab-qe 应用, 尝试自动克隆... (目录: $APPS_DIR/aiidalab-qe)" >&2
    if command -v git >/dev/null 2>&1; then
      if git clone --depth=1 https://github.com/aiidalab/aiidalab-qe.git "$APPS_DIR/aiidalab-qe" 2>&1; then
        echo "[INFO] aiidalab-qe 克隆完成" >&2
      else
        echo "[WARN] 克隆 aiidalab-qe 失败, 请手动安装 (可能网络受限)" >&2
      fi
    else
      echo "[WARN] 未找到 git, 跳过自动克隆" >&2
    fi
  else
    echo "[INFO] 检测到已存在 aiidalab-qe 应用目录, 跳过引导" >&2
  fi
}

if [[ $BOOTSTRAP_QE -eq 1 ]]; then
  bootstrap_qe_app
fi

mkdir -p "$LOG_DIR"

# 复用端口文件
if [[ $REUSE -eq 1 && -f "$PORT_FILE" ]]; then
  SAVED_PORT=$(cat "$PORT_FILE" 2>/dev/null || true)
  if [[ -n "$SAVED_PORT" && "$SAVED_PORT" != "$PORT" ]]; then
    echo "[INFO] 读取历史端口: $SAVED_PORT (覆盖请求端口 $PORT)" >&2
    PORT="$SAVED_PORT"
  fi
fi

# 强制重启: 清理 tmux 和旧 jupyter
if [[ $FORCE_RESTART -eq 1 ]]; then
  if tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
    echo "[INFO] 强制重启: 终止已有 tmux 会话 $SESSION_NAME" >&2
    tmux kill-session -t "$SESSION_NAME" || true
  fi
  OLD_PIDS=$(pgrep -f "jupyter.*--port=$PORT" || true)
  if [[ -n "$OLD_PIDS" ]]; then
    echo "[INFO] 清理旧 Jupyter 进程: $OLD_PIDS" >&2
    kill $OLD_PIDS 2>/dev/null || true
  fi
fi

# 复用已有进程
if [[ $REUSE -eq 1 && $FORCE_RESTART -eq 0 ]]; then
  if existing_jupyter | grep -q "--port=$PORT"; then
    echo "[INFO] 发现已有 Jupyter 进程 (端口: $PORT) 复用" >&2
    print_access_info; exit 0
  fi
  ANY_LINE=$(list_jupyter_any | head -n1 || true)
  if [[ -n "$ANY_LINE" ]]; then
    DETECTED_PORT=$(echo "$ANY_LINE" | sed -n 's/.*--port=\([0-9]\+\).*/\1/p')
    if [[ -n "$DETECTED_PORT" ]]; then
      echo "[INFO] 发现其他运行中的 Jupyter (端口: $DETECTED_PORT) - 使用其端口复用" >&2
      PORT="$DETECTED_PORT"; echo "$PORT" > "$PORT_FILE" 2>/dev/null || true
      print_access_info; exit 0
    fi
  fi
fi

JUPYTER_CMD=$(resolve_jupyter_cmd)

if [[ $USE_TMUX -eq 1 ]]; then
  need_cmd tmux
  if tmux list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}:"; then
    if [[ $FORCE_RESTART -eq 1 ]]; then
      echo "[WARN] 会话残留再清理" >&2
      tmux kill-session -t "$SESSION_NAME" || true
    else
      if [[ -f "$PORT_FILE" ]]; then SAVED_PORT=$(cat "$PORT_FILE" 2>/dev/null || true); [[ -n "$SAVED_PORT" ]] && PORT="$SAVED_PORT"; fi
      echo "[INFO] tmux 会话已存在: $SESSION_NAME (复用, 端口: $PORT)" >&2
      print_access_info; echo "附加会话: tmux attach -t $SESSION_NAME"; exit 0
    fi
  fi
  echo "[INFO] 创建 tmux 会话: $SESSION_NAME" >&2
  tmux new-session -d -s "$SESSION_NAME" "echo '[tmux] 启动 Jupyter (端口 $PORT)...'; $JUPYTER_CMD 2>&1 | tee -a '$LOG_DIR/aiidalab_qe_${PORT}.log'"
  sleep 2
  echo "$PORT" > "$PORT_FILE" 2>/dev/null || true
  print_access_info; echo "附加会话: tmux attach -t $SESSION_NAME"; exit 0
fi

if [[ $USE_NOHUP -eq 1 ]]; then
  echo "[INFO] 以 nohup 后台运行" >&2
  nohup bash -c "$JUPYTER_CMD" >"$LOG_DIR/aiidalab_qe_${PORT}.log" 2>&1 &
  disown || true; sleep 2
  echo "$PORT" > "$PORT_FILE" 2>/dev/null || true
  print_access_info; exit 0
fi

echo "[INFO] 前台启动: $JUPYTER_CMD" >&2
print_access_info
echo "$PORT" > "$PORT_FILE" 2>/dev/null || true
exec bash -c "$JUPYTER_CMD"
