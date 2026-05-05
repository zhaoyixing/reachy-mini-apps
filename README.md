# Reachy Mini × 豆包端到端语音对话 App

基于火山引擎 `volc.speech.dialog` 端到端实时语音大模型，让 Reachy Mini 机器人实现像豆包手机 App 一样的自然对话体验。

---

## 功能特性

- **一路 WebSocket，全链路端到端**：ASR → LLM（豆包） → TTS 全部在一条连接内完成，延迟极低
- **实时双向对话**：说话即触发，无需按键，连续多轮对话
- **情绪动作联动**：豆包回复中含情绪标签（如 `[开心]`），自动映射为机器人预录情绪动作（18 种）
- **语音驱动头部晃动**：TTS 播放时，音量/节奏实时驱动头部 Sway 动画，自然有生命感
- **声源定向**：倾听时通过 DoA（Direction of Arrival）自动转向说话人
- **视觉理解**：集成摄像头，用豆包方舟视觉模型或 Gemini 分析画面，支持"你看到了什么"类问答
- **自动重连**：WebSocket 断线后自动重连，长时间运行无需干预

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      Reachy Mini 机器人                      │
│                                                             │
│  麦克风 ──► AudioCapture ──► AudioBuffer (20ms 对齐)         │
│                                  │                          │
│                                  ▼                          │
│                        DoubaoRealtimeClient                 │
│                        ┌──────────────────┐                 │
│                        │  WebSocket (WSS) │◄──── 火山引擎   │
│                        │  V3 Binary Proto │                 │
│                        │                 │  ASR+LLM+TTS     │
│                        │  send_audio()   │  端到端实时       │
│                        │  on_event()     │  语音大模型       │
│                        │  on_audio()     │                  │
│                        └──────────────────┘                 │
│                            │          │                     │
│               JSON 事件    │          │  PCM 音频           │
│                            ▼          ▼                     │
│                     MotionController  扬声器播放             │
│                     ┌─────────────┐                         │
│                     │ 状态机       │                         │
│                     │ idle        │                         │
│                     │ listening   │──► DoA 声源转向         │
│                     │ thinking    │──► 静止等待             │
│                     │ speaking    │──► 情绪动作 + Sway      │
│                     └─────────────┘                         │
│                                                             │
│  摄像头 ──► VisionAnalyzer ──► 豆包方舟 / Gemini ──► 注入对话│
└─────────────────────────────────────────────────────────────┘
```

---

## 模块说明

### `main.py` — 主程序 & 编排层

- 继承 `ReachyMiniApp`，接管 Reachy Mini SDK 生命周期
- 管理音频采集线程，将 float32 stereo → int16 mono 后送入豆包客户端
- 处理豆包回调：解析情绪标签、触发动作、推送 TTS 音频到扬声器
- 视觉线程：定期抓帧 → 调用 `vision.py` → 将描述注入对话（event 501）
- 优雅关闭：Ctrl+C / 检测到"再见"意图时安全退出

### `doubao_client.py` — 豆包 Realtime WebSocket 客户端

| 函数 / 类 | 说明 |
|-----------|------|
| `DoubaoRealtimeClient` | 核心客户端，管理连接、鉴权、会话 |
| `_build_json_frame()` | 构造 V3 控制帧（gzip 压缩 JSON） |
| `_build_audio_frame()` | 构造 V3 音频帧（gzip 压缩 PCM） |
| `_parse_frame()` | 统一解析所有服务端响应帧 |
| `send_audio()` | 40ms 间隔批量发送音频（减少 QPM 占用） |

**V3 协议帧结构（自研逆向）：**

```
控制帧:  [Header 4B] [EventType 4B] [SID_len 4B] [SID] [Payload_len 4B] [gzip(JSON)]
音频帧:  [Header 4B] [EventType 4B] [SID_len 4B] [SID] [Payload_len 4B] [gzip(PCM)]
响应帧:  [Header 4B] [Flags] [EventType 4B] [SID_len 4B] [SID] [Payload_len 4B] [gzip(data)]
```

**关键事件码：**

| 事件 | 含义 |
|------|------|
| 100 | StartSession（发起会话，配置模型/TTS/人格） |
| 102 | FinishSession |
| 150 | SessionStarted（会话就绪） |
| 200/201 | 用户开始/结束说话（进入 listening 状态） |
| 450 | LLM 思考中（进入 thinking 状态） |
| 459 | 回复结束（回到 idle） |

### `motion_controller.py` — 动作状态机

- **4 种状态**：`idle`、`listening`、`thinking`、`speaking`，30 Hz 控制循环
- **情绪动作**：集成 `pollen-robotics/reachy-mini-emotions-library`，18 种预录动作与豆包情绪标签自动映射
- **Sway 算法**：调用 `SwayRollRT`，将 TTS 音频的音量/节奏实时转化为头部 6DOF 晃动偏移
- **DoA 声源定向**：倾听时调用 `robot.media.get_DoA()` 获取声源角度，驱动头部偏转

**情绪映射表（部分）：**

| 豆包标签 | 动作库名称 |
|----------|-----------|
| `[开心]` | cheerful1 |
| `[惊讶]` | amazed1 |
| `[困惑]` | confused1 |
| `[大笑]` | laughing1 |
| … | … 共 18 种 |

### `vision.py` — 视觉理解模块

- 输入：OpenCV BGR 帧（来自 `reachy_mini.media`）
- 流程：帧 → JPEG 压缩 → base64 → Vision API → 描述文本
- 后端优先级：**豆包方舟视觉模型**（国内） > **Gemini 2.0 Flash**（备用）
- 描述文本以 `event 501`（TextQuery）形式注入豆包对话

### `speech_tapper.py` — 语音 Sway 算法

- `SwayRollRT`：流式音频处理，输出每帧的 6DOF 偏移量（x/y/z mm + roll/pitch/yaw °）
- 基于音量包络（RMS dBFS）和多频率正弦叠加，模拟自然说话时的身体律动
- 复用自 [pollen-robotics/reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app)（Apache 2.0）

---

## 环境配置（`.env`）

```ini
# 必填：火山引擎豆包 API Key
DOUBAO_API_KEY=your-api-key-here

# 可选：TTS 声音
DOUBAO_SPEAKER=zh_female_wanwanxiaohe_moon_bigtts

# 可选：视觉模型（豆包方舟）
ARK_API_KEY=your-ark-key
DOUBAO_VISION_ENDPOINT=doubao-1-5-vision-pro-32k-250115

# 可选：视觉模型（Gemini 备用）
GEMINI_API_KEY=your-gemini-key
```

---

## 运行方式

```bash
# 在机器人上执行（需 daemon 已运行）
cd ~/reachy_doubao_voice_app
/venvs/apps_venv/bin/python main.py
```

---

## 依赖

- `reachy-mini` SDK（预装于 `/venvs/apps_venv/`）
- `websocket-client`、`python-dotenv`、`numpy`
- 火山引擎账号 + `volc.speech.dialog` 服务开通
