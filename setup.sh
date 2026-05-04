#!/bin/bash
# setup.sh —— 在 Reachy Mini 上安装依赖并运行
# 使用系统预置的 /venvs/apps_venv/（与其他 Reachy app 保持一致）
set -e

PYTHON="/venvs/apps_venv/bin/python"
PIP="/venvs/apps_venv/bin/pip"

echo "🔧 检查 Python 环境..."
$PYTHON --version

echo ""
echo "📦 安装额外依赖到 /venvs/apps_venv/ ..."
$PIP install websocket-client python-dotenv numpy soundfile

echo ""
echo "🚀 启动豆包语音对话 APP..."
echo "   按 Ctrl+C 停止"
echo ""

# 与 reachy-mini-conversation-app 保持一致：设置 no_proxy
export no_proxy=localhost,127.0.0.1,192.168.1.14,reachy-mini.local
export NO_PROXY=localhost,127.0.0.1,192.168.1.14,reachy-mini.local

exec $PYTHON main.py
