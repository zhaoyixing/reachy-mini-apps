#!/bin/bash
# deploy.sh —— 把豆包语音 APP 部署到 Reachy Mini
set -e

ROBOT_HOST="${ROBOT_HOST:-reachy-mini.local}"
ROBOT_USER="${ROBOT_USER:-pollen}"
REMOTE_DIR="/home/pollen/reachy_doubao_voice_app"

echo "🤖 部署目标: ${ROBOT_USER}@${ROBOT_HOST}"
echo "📁 远程目录: ${REMOTE_DIR}"

# 1. 创建远程目录
ssh -o StrictHostKeyChecking=no ${ROBOT_USER}@${ROBOT_HOST} "mkdir -p ${REMOTE_DIR}"

# 2. 同步代码（排除 .env，防止覆盖远程配置）
rsync -avz --exclude='.env' --exclude='__pycache__' --exclude='*.pyc' \
  "$(dirname "$0")/" \
  ${ROBOT_USER}@${ROBOT_HOST}:${REMOTE_DIR}/

# 3. 同步 .env（如果不存在才复制）
ssh ${ROBOT_USER}@${ROBOT_HOST} \
  "[ -f ${REMOTE_DIR}/.env ] || cp ${REMOTE_DIR}/.env.example ${REMOTE_DIR}/.env 2>/dev/null || true"

echo "✅ 代码已同步到机器人"
echo ""
echo "接下来请 SSH 到机器人并安装依赖："
echo "  ssh ${ROBOT_USER}@${ROBOT_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  ./setup.sh"
