# Reachy Mini × 豆包语音对话 APP —— 项目记忆

> 本文件记录项目的开发历史、关键决策、已知问题和待办事项。
> 最后更新：2026-05-04

---

## 1. 项目概述

基于 **pollen-robotics/reachy_mini_conversation_app**（Apache 2.0）改造，将原有的 OpenAI/Gemini Realtime API 替换为**豆包端到端实时语音大模型 API**（火山引擎）。

机器人：Pollen Robotics Reachy Mini（树莓派 + 头部/天线电机 + 摄像头 + 麦克风）
运行环境：/venvs/apps_venv/bin/python（Python 3.12）

---

## 2. 已实现功能

### 核心语音对话
- 麦克风采集 → 豆包 Realtime ASR → 大模型 → TTS → 扬声器播放
- 流式 OGG Opus 音频解码
- 数字增益 2.0 倍（解决音量过小问题）
- ASR 结束检测优化：`end_smooth_window_ms` 1500ms → 200ms
- 音频发送间隔优化：100ms → 40ms

### 机器人动作
- **状态机动画**：idle / listening / thinking / speaking 四种状态对应不同头部+天线动作
- **语音驱动摇头（Sway）**：根据 TTS 音频响度实时驱动头部晃动
- **情绪动作**：集成官方 `reachy-mini-emotions-library`，在 speaking 时播放对应情绪动画
- 情绪标签 system prompt：要求豆包在回复开头输出 `[情绪名]`（如 `[开心]` `[害羞]`）

### 视觉分析（多模态尝试）
- 后台线程每 1.5 秒抓取摄像头帧
- 支持豆包方舟视觉模型（`doubao-1-5-vision-pro-32k-250115`）
- 支持 Gemini 2.0 Flash 作为备选
- **代码保留但存在架构限制**（详见第 4 节）

---

## 3. 技术栈

| 组件 | 说明 |
|------|------|
| Python 3.12 | 运行环境 |
| reachy-mini SDK | 机器人控制、Zenoh 通信、媒体后端（GStreamer） |
| 豆包 Realtime API V3 | 端到端语音对话（wss://openspeech.bytedance.com） |
| 豆包方舟 API | 视觉理解（OpenAI-compatible 格式） |
| websocket-client | WebSocket 连接 |
| numpy / soundfile | 音频处理 |
| Pillow | 图像编码（视觉模块） |

---

## 4. 已知问题与限制

### 4.1 视觉问答无法真正联动（核心限制）

**问题描述**：
- 视觉分析模块本身工作正常（能正确识别画面内容）
- 但无法让豆包 Realtime 语音模型在**当前轮次**的回答中结合画面信息

**根本原因**：
- 豆包 Realtime API 是**流式语音到语音**架构
- 用户说完话后，模型立刻开始生成回答（thinking → speaking）
- 通过 `chat_text_query`（event=501）插入的画面描述，会被协议层当作**新的用户轮次**，而不是"补充当前问题的上下文"
- 这导致模型要么忽略画面描述，要么把它当作第二轮独立问题来回答

**已尝试的方案**：
1. thinking 阶段注入 → 太晚，模型已开始生成
2. listening 阶段注入 → 画面描述和用户语音变成两条独立消息，模型仍无法正确关联
3. system prompt 指示模型"你能看见画面" → 无效，因为模型本身没有视觉输入通道

**结论**：
- 当前代码保留（`vision.py` + `main.py` 中的视觉线程），作为基础设施
- 要真正实现"拿着东西问这是什么"，需要改用**双模型协作架构**（见第 6 节待办）

### 4.2 daemon 长时间运行后卡死

- reachy-mini-daemon 运行约 2-3 小时后 CPU 飙满（57%+），HTTP API 无响应
- 临时解决：`sudo systemctl restart reachy-mini-daemon.service`
- 根本原因待查（可能是 wireless 模式的某个服务内存泄漏）

### 4.3 情绪标签遵循率

- system prompt 要求豆包输出 `[情绪名]` 标签，但模型不一定严格遵循
- 已增加兜底机制：每次 speaking 自动随机播放一个情绪动作

---

## 5. 关键配置

`.env` 文件：

```bash
# 豆包语音对话（必需）
DOUBAO_API_KEY=...
DOUBAO_SPEAKER=zh_female_wanwanxiaohe_moon_bigtts

# 豆包方舟视觉模型（视觉模块用）
DOUBAO_ARK_API_KEY=...
DOUBAO_VISION_ENDPOINT=doubao-1-5-vision-pro-32k-250115

# Gemini 备选（视觉模块用）
# GEMINI_API_KEY=...
```

**运行命令**：
```bash
cd /home/pollen/reachy_doubao_voice_app && /venvs/apps_venv/bin/python main.py
```

---

## 6. 待办事项（TODO）

### 高优先级
1. **视觉问答的真正实现**
   - 方案：双模型协作
   - 豆包 Realtime 负责语音交互（听 + 说）
   - 当用户问视觉相关问题时，暂停 Realtime 生成
   - 抓图 + 用户 ASR 文本 → 豆包方舟视觉+文本模型 → 文本回答
   - 文本回答 → 本地 TTS 或豆包 TTS → 语音播报
   - 挑战：如何获取 Realtime 的 ASR 文本（当前 API 不返回）

2. **连接 Mac 执行指令**
   - 用户提到 "OpenClaw"，可能是 Open Interpreter
   - 架构：Mac 端运行轻量 HTTP 服务，Reachy Mini 通过 HTTP 发送指令
   - 或集成 Open Interpreter API 模式

3. **长思考 / 复杂任务执行**
   - 在 system prompt 中加入 Chain-of-Thought 指令
   - 让模型先拆解任务、再逐步执行

### 中优先级
4. **daemon 稳定性**
   - 排查 wireless 模式下长时间运行的 CPU 飙满问题
   - 或设置定时自动重启

5. **情绪动作精准匹配**
   - 优化情绪标签解析，提高豆包遵循率
   - 或基于语义分析（NLP）替代标签匹配

### 低优先级
6. **Gradio Web UI**
   - 当前 `dont_start_webserver = True`
   - 可以开启用于调试和监控

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `main.py` | APP 主入口，语音对话循环，视觉联动 |
| `doubao_client.py` | 豆包 Realtime V3 WebSocket 客户端 |
| `motion_controller.py` | 机器人动作状态机，情绪动作播放 |
| `vision.py` | 视觉分析模块（豆包方舟 / Gemini） |
| `speech_tapper.py` | 语音驱动的头部晃动算法 |
| `.env` | API Key 和配置 |
| `setup.sh` | 一键安装依赖并启动 |
| `deploy.sh` | 部署脚本（rsync 到机器人） |
| `docs/MEMORY.md` | 本文件 |

---

## 8. 对话历史摘要

### 2026-05-04 开发记录

**启动问题**
- 报错 `zenoh.ZError: Unable to connect to tcp/localhost:7447`
- 原因：daemon 以 `--no-autostart` 运行，backend 未启动
- 解决：通过 API `POST /api/daemon/start?wake_up=true` 启动 backend

**依赖问题**
- 报错 `ModuleNotFoundError: No module named 'numpy'`
- 原因：用了系统 Python（/usr/bin/python）而非 /venvs/apps_venv/bin/python
- 解决：使用 apps_venv 的 Python 运行

**音量优化**
- 系统 PCM 音量 62% → `amixer set PCM 60` 调到 100%
- 代码中增加数字增益：`np.clip(stereo * 2.0, -1.0, 1.0)`

**延迟优化**
- `end_smooth_window_ms` 1500 → 200
- 音频发送间隔 100ms → 40ms

**情绪动作**
- 集成 `reachy-mini-emotions-library`
- system prompt 要求豆包输出 `[情绪名]` 标签
- 增加兜底机制：无标签时随机播放情绪动作

**视觉尝试**
- 单独测试视觉分析成功
- 集成到对话流程失败（Realtime API 架构限制）
- 代码保留，待后续改用双模型架构

---

*本文件由开发助手整理，用于保留项目上下文和对话记忆。*
