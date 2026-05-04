"""
Reachy Mini × 豆包端到端语音大模型 —— 对话 APP
================================================
参考架构：pollen-robotics/reachy_mini_conversation_app（Apache 2.0）
核心替换：OpenAI/Gemini Realtime API → 豆包 Realtime API（火山引擎）

运行方式：
    1. 填入 .env 中的火山引擎凭证
    2. 在机器人上（或同一网络）运行：
       python main.py

环境要求：
    - reachy-mini SDK 已安装
    - 机器人 daemon 正在运行
    - 网络可访问火山引擎 wss://openspeech.bytedance.com
"""
from __future__ import annotations
import os
import sys
import time
import signal
import logging
import threading
from typing import Optional

import io
import re
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.media.media_manager import MediaManager, MediaBackend

from doubao_client import DoubaoRealtimeClient, CHUNK_BYTES, CHUNK_SAMPLES
from motion_controller import MotionController
import vision

# --------------------------------------------------------------------------- #
# 日志配置
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reachy_doubao")

# --------------------------------------------------------------------------- #
# 音频工具
# --------------------------------------------------------------------------- #

class AudioBuffer:
    """把不固定大小的音频 chunk 整理成固定大小的包（默认 20ms / 640 bytes）。"""

    def __init__(self, chunk_bytes: int = CHUNK_BYTES) -> None:
        self.chunk_bytes = chunk_bytes
        self._buf = bytearray()

    def feed(self, pcm_int16: np.ndarray) -> list[bytes]:
        """传入 int16 数组，返回 List[chunk_bytes bytes]。"""
        self._buf.extend(pcm_int16.tobytes())
        packets = []
        while len(self._buf) >= self.chunk_bytes:
            packets.append(bytes(self._buf[:self.chunk_bytes]))
            self._buf = self._buf[self.chunk_bytes:]
        return packets

    def drain(self) -> Optional[bytes]:
        """取走剩余数据（可能不足 20ms）。"""
        if self._buf:
            data = bytes(self._buf)
            self._buf = bytearray()
            return data
        return None


def f32_stereo_to_i16_mono(sample: np.ndarray) -> np.ndarray:
    """
    Reachy Mini 录音格式: float32 stereo (N, 2) → 豆包输入: int16 mono (N,)
    """
    # 先平均为 mono
    mono_f32 = sample.mean(axis=1)
    # 裁剪到 [-1, 1] 防止溢出
    mono_f32 = np.clip(mono_f32, -1.0, 1.0)
    # 转 int16
    return (mono_f32 * 32767.0).astype(np.int16)


def i16_mono_to_f32_stereo(pcm_bytes: bytes) -> np.ndarray:
    """
    豆包输出: int16 mono → Reachy Mini 播放格式: float32 stereo (N, 2)
    """
    arr_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    arr_f32 = arr_i16.astype(np.float32) / 32767.0
    # 复制为 stereo (N, 2)
    return np.column_stack([arr_f32, arr_f32])


class IncrementalOggDecoder:
    """增量 OGG Opus 解码器，支持流式追加和播放。"""

    def __init__(self, target_sr: int = 16000) -> None:
        self.target_sr = target_sr
        self._buffer = bytearray()
        self._decoded_samples = 0

    def reset(self) -> None:
        """重置缓冲区（TTS 开始时调用）。"""
        self._buffer = bytearray()
        self._decoded_samples = 0

    def feed(self, ogg_bytes: bytes) -> np.ndarray | None:
        """追加 OGG 数据，返回新解码的 float32 stereo 样本（如果有）。"""
        if not ogg_bytes:
            return None
        self._buffer.extend(ogg_bytes)
        try:
            buf = io.BytesIO(bytes(self._buffer))
            data, sr = sf.read(buf, dtype="float32")
        except Exception:
            return None

        if data.ndim == 2:
            data = data.mean(axis=1)

        if len(data) <= self._decoded_samples:
            return None

        new_data = data[self._decoded_samples:]
        self._decoded_samples = len(data)

        # 重采样到 target_sr
        if sr == 24000 and self.target_sr == 48000:
            x = np.arange(len(new_data))
            x_new = np.arange(len(new_data) * 2) / 2.0
            new_data = np.interp(x_new, x, new_data)
        elif sr != self.target_sr:
            ratio = self.target_sr / sr
            x = np.arange(len(new_data))
            x_new = np.arange(int(len(new_data) * ratio)) / ratio
            new_data = np.interp(x_new, x, new_data)

        return np.column_stack([new_data, new_data]).astype(np.float32)

    def flush(self) -> np.ndarray | None:
        """尝试解码剩余所有数据。"""
        if not self._buffer:
            return None
        try:
            buf = io.BytesIO(bytes(self._buffer))
            data, sr = sf.read(buf, dtype="float32")
        except Exception:
            return None

        if data.ndim == 2:
            data = data.mean(axis=1)

        if len(data) <= self._decoded_samples:
            return None

        new_data = data[self._decoded_samples:]
        self._decoded_samples = len(data)

        if sr == 24000 and self.target_sr == 48000:
            x = np.arange(len(new_data))
            x_new = np.arange(len(new_data) * 2) / 2.0
            new_data = np.interp(x_new, x, new_data)
        elif sr != self.target_sr:
            ratio = self.target_sr / sr
            x = np.arange(len(new_data))
            x_new = np.arange(int(len(new_data) * ratio)) / ratio
            new_data = np.interp(x_new, x, new_data)

        return np.column_stack([new_data, new_data]).astype(np.float32)


# --------------------------------------------------------------------------- #
# APP 主类
# --------------------------------------------------------------------------- #

class DoubaoVoiceApp(ReachyMiniApp):
    """Reachy Mini 豆包语音对话 APP。"""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = True  # 先不做 Gradio UI，命令行运行

    def __init__(self, running_on_wireless: bool = False) -> None:
        super().__init__(running_on_wireless=running_on_wireless)
        # Wireless 模式下 daemon 在本地但 zenoh 未启动，强制使用 network 连接
        if running_on_wireless:
            self.daemon_on_localhost = False
        self._stop_event: Optional[threading.Event] = None
        self._robot: Optional[ReachyMini] = None
        self._media: Optional[MediaManager] = None
        self._motion: Optional[MotionController] = None
        self._client: Optional[DoubaoRealtimeClient] = None

        # 运行标志
        self._running = False
        self._audio_thread: Optional[threading.Thread] = None

        # 多模态视觉
        self._gemini_api_key: Optional[str] = None
        self._doubao_ark_key: Optional[str] = None
        self._doubao_vision_ep: Optional[str] = None
        self._last_vision_desc: Optional[str] = None
        self._vision_last_trigger: float = 0.0
        self._vision_lock = threading.Lock()
        self._vision_thread: Optional[threading.Thread] = None
        self._vision_stop = threading.Event()
        self._vision_injected: bool = False

    # ------------------------------------------------------------------ #
    # ReachyMiniApp 入口
    # ------------------------------------------------------------------ #
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        logger.info("=" * 60)
        logger.info("Reachy Mini 豆包语音对话 APP 启动")
        logger.info("=" * 60)

        self._stop_event = stop_event
        self._robot = reachy_mini

        # 读取环境变量
        load_dotenv()
        api_key = os.getenv("DOUBAO_API_KEY", "").strip()
        app_id = os.getenv("DOUBAO_APP_ID", "").strip()
        access_key = os.getenv("DOUBAO_ACCESS_KEY", "").strip()
        app_key = os.getenv("DOUBAO_APP_KEY", "").strip()
        model = os.getenv("DOUBAO_MODEL", "2.2.0.0").strip()
        speaker = os.getenv("DOUBAO_SPEAKER", "zh_female_wanwanxiaohe_moon_bigtts").strip()
        self._gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        self._doubao_ark_key = os.getenv("DOUBAO_ARK_API_KEY", "").strip() or None
        self._doubao_vision_ep = os.getenv("DOUBAO_VISION_ENDPOINT", "").strip() or None
        has_vision = bool((self._doubao_ark_key and self._doubao_vision_ep) or self._gemini_api_key)

        if self._doubao_ark_key and self._doubao_vision_ep:
            logger.info("豆包方舟视觉分析已启用（endpoint=%s）", self._doubao_vision_ep)
        elif self._gemini_api_key:
            logger.info("Gemini 视觉分析已启用")
        else:
            logger.info("视觉分析未启用（如需多模态，请在 .env 中添加 DOUBAO_ARK_API_KEY + DOUBAO_VISION_ENDPOINT，或 GEMINI_API_KEY）")

        # 启动后台视觉线程：像人眼一样持续"睁着"
        if has_vision:
            self._vision_stop.clear()
            self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
            self._vision_thread.start()
            logger.info("后台视觉线程已启动")

        # 新版优先：单 API Key；兼容旧版三参数
        if api_key:
            logger.info("使用新版鉴权（DOUBAO_API_KEY）")
        elif app_id and access_key and app_key:
            logger.info("使用旧版鉴权（DOUBAO_APP_ID + DOUBAO_ACCESS_KEY + DOUBAO_APP_KEY）")
        else:
            logger.error(
                "火山引擎凭证不完整！请检查 .env 文件。\n"
                "新版只需配置: DOUBAO_API_KEY\n"
                "旧版需配置 : DOUBAO_APP_ID / DOUBAO_ACCESS_KEY / DOUBAO_APP_KEY"
            )
            return

        # 初始化 Media（音频采集/播放）
        backend_str = os.getenv("MEDIA_BACKEND", "default").strip().lower()
        if backend_str == "webrtc":
            backend = MediaBackend.WEBRTC
        elif backend_str == "no_media":
            backend = MediaBackend.NO_MEDIA
        else:
            backend = MediaBackend.DEFAULT

        logger.info("Media backend: %s", backend.value)
        try:
            self._media = reachy_mini.media
        except Exception as exc:
            logger.error("无法初始化媒体: %s", exc)
            return

        self._media.start_recording()
        self._media.start_playing()
        logger.info("音频采集与播放已启动")

        # 初始化动作控制器
        self._motion = MotionController(reachy_mini)
        self._motion.start()

        # 初始化豆包客户端
        self._client = DoubaoRealtimeClient(
            api_key=api_key or None,
            app_id=app_id or None,
            access_key=access_key or None,
            app_key=app_key or None,
            model=model,
            speaker=speaker,
            on_event=self._on_doubao_event,
            on_audio=self._on_doubao_audio,
            on_state=self._on_doubao_state,
        )
        self._client.connect()

        # 等待连接就绪（最多 10 秒）
        for _ in range(50):
            if self._client.connected:
                break
            time.sleep(0.2)
        if not self._client.connected:
            logger.error("无法连接到豆包 Realtime API，请检查网络和凭证")
            self._shutdown()
            return

        logger.info("豆包 Realtime API 已连接，模型: %s", model)
        logger.info('开始对话！对机器人说话吧，说"再见"或按 Ctrl+C 结束。')

        # 启动音频采集线程
        self._running = True
        self._audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self._audio_thread.start()

        # 主线程等待停止信号
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------ #
    # 音频采集线程
    # ------------------------------------------------------------------ #
    def _audio_capture_loop(self) -> None:
        """持续从麦克风读取音频，发送给豆包。"""
        buf = AudioBuffer()
        while self._running and not self._stop_event.is_set():
            try:
                sample = self._media.get_audio_sample()
            except Exception as exc:
                logger.debug("get_audio_sample error: %s", exc)
                time.sleep(0.01)
                continue

            if sample is None or sample.size == 0:
                time.sleep(0.005)
                continue

            # 格式转换: float32 stereo → int16 mono
            mono_i16 = f32_stereo_to_i16_mono(sample)
            packets = buf.feed(mono_i16)
            for pkt in packets:
                if self._client:
                    self._client.send_audio(pkt)

        # 线程退出时清空剩余音频
        remainder = buf.drain()
        if remainder and self._client:
            self._client.send_audio(remainder)

    # ------------------------------------------------------------------ #
    # 豆包回调
    # ------------------------------------------------------------------ #
    _VALID_EMOTIONS = {
        "开心", "害羞", "惊讶", "生气", "大笑", "难过", "困惑",
        "感激", "无聊", "害怕", "平静", "好奇", "沮丧", "轻蔑",
        "厌恶", "不耐烦", "兴奋", "疲惫",
    }

    def _extract_emotion(self, text: str) -> Optional[str]:
        """从文本中提取 [情绪名] 标签。"""
        m = re.search(r"\[([^\]]+)\]", text)
        if m:
            emotion = m.group(1)
            if emotion in self._VALID_EMOTIONS:
                return emotion
        return None

    def _on_doubao_event(self, event: dict) -> None:
        """收到 JSON 控制事件。"""
        event_type = event.get("event", "Unknown")
        event_num = event.get("_event_type")
        payload = event.get("payload", {})
        extra = event.get("extra", {})

        # 调试：打印所有含 text 的事件，确认豆包返回结构
        text = payload.get("text") or payload.get("content") or extra.get("text") or extra.get("content")
        if text and isinstance(text, str):
            logger.info("[豆包文本] event=%s text=%s", event_num, text[:80])
            if self._motion:
                emotion = self._extract_emotion(text)
                if emotion:
                    logger.info("[豆包情绪] 检测到: %s", emotion)
                    self._motion.trigger_emotion(emotion)

        # TTS 开始 (event=350): 重置音频解码器
        if event_num == 350:
            if hasattr(self, "_ogg_decoder"):
                self._ogg_decoder.reset()

        # TTSEnded (event=359) 时检查退出意图并刷新剩余音频
        if event_num == 359:
            if hasattr(self, "_ogg_decoder"):
                remaining = self._ogg_decoder.flush()
                if remaining is not None and self._media:
                    try:
                        self._media.push_audio_sample(remaining)
                    except Exception as exc:
                        logger.warning("播放剩余音频失败: %s", exc)
            extra = event.get("extra", {})
            if extra.get("user_query_exit"):
                logger.info("检测到用户退出意图，准备关闭...")
                if self._stop_event:
                    self._stop_event.set()

    def _on_doubao_audio(self, audio_bytes: bytes) -> None:
        """收到合成音频（OGG Opus），流式解码、播放并驱动动作。"""
        if not audio_bytes:
            return

        # 懒创建增量解码器
        # Reachy Mini 音频硬件采样率为 16000Hz，必须与之一致
        if not hasattr(self, "_ogg_decoder"):
            self._ogg_decoder = IncrementalOggDecoder(target_sr=16000)

        # 流式解码并播放新样本
        try:
            stereo = self._ogg_decoder.feed(audio_bytes)
            if stereo is not None and self._media:
                # 数字增益：放大 2 倍并限幅，防止爆音
                stereo = np.clip(stereo * 2.0, -1.0, 1.0)
                self._media.push_audio_sample(stereo)
        except Exception as exc:
            logger.warning("音频播放失败: %s", exc)

        # 语音驱动动作（需要 int16 数据）
        # 动作驱动不需要流式，累积到最后再驱动也可以
        if self._motion:
            try:
                buf = io.BytesIO(audio_bytes)
                data, sr = sf.read(buf, dtype="int16")
                if data.ndim == 2:
                    data = data.mean(axis=1).astype(np.int16)
                self._motion.feed_speech_audio(data, sr=sr)
            except Exception as exc:
                logger.debug("动作驱动失败: %s", exc)

    def _vision_loop(self) -> None:
        """后台持续抓图分析，像人眼一样保持最新画面记忆。"""
        logger.info("视觉线程启动，每 1.5 秒分析一帧")
        while not self._vision_stop.is_set():
            time.sleep(1.5)
            if self._media is None:
                continue
            try:
                frame = self._media.get_frame()
            except Exception:
                continue
            if frame is None:
                continue
            desc = vision.analyze_frame(
                frame,
                prompt="用一句话描述这张图片里的主要物体。",
                doubao_ark_key=self._doubao_ark_key,
                doubao_vision_endpoint=self._doubao_vision_ep,
                gemini_api_key=self._gemini_api_key,
            )
            if desc:
                with self._vision_lock:
                    self._last_vision_desc = desc

    def _inject_vision_context(self) -> None:
        """把最新的视觉描述注入豆包对话。"""
        with self._vision_lock:
            desc = self._last_vision_desc
        if not desc or not self._client:
            return
        query = f"【我看到的画面】{desc}。请结合这个画面来回答用户的问题。"
        try:
            self._client.chat_text_query(query)
            logger.info("[Vision注入] %s", query[:100])
        except Exception as exc:
            logger.warning("Vision 注入失败: %s", exc)

    def _on_doubao_state(self, state: str) -> None:
        """连接/对话状态变化。"""
        logger.info("[豆包状态] %s", state)
        if self._motion:
            if state in ("listening", "thinking", "speaking", "idle"):
                self._motion.set_state(state)
            elif state == "disconnected":
                self._motion.set_state("idle")

        # 多模态视觉联动：listening 开始时注入画面描述，确保 ASR 完成前已在上下文
        has_vision_backend = bool(
            (self._doubao_ark_key and self._doubao_vision_ep) or self._gemini_api_key
        )
        if state == "listening" and has_vision_backend and not self._vision_injected:
            self._inject_vision_context()
            self._vision_injected = True
        elif state in ("speaking", "idle"):
            self._vision_injected = False

    # ------------------------------------------------------------------ #
    # 优雅关闭
    # ------------------------------------------------------------------ #
    def _shutdown(self) -> None:
        logger.info("正在关闭 APP...")
        self._running = False

        if self._client:
            self._client.close()
            self._client = None

        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)

        if self._motion:
            self._motion.stop()
            self._motion = None

        if self._vision_stop:
            self._vision_stop.set()
        if self._vision_thread and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=2.0)

        if self._media:
            try:
                self._media.stop_recording()
                self._media.stop_playing()
                self._media.close()
            except Exception as exc:
                logger.debug("Media close error: %s", exc)
            self._media = None

        if self._robot:
            try:
                self._robot.client.disconnect()
            except Exception as exc:
                logger.debug("Robot disconnect error: %s", exc)
            self._robot = None

        logger.info("APP 已安全关闭")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main() -> None:
    # 支持 Ctrl+C 优雅退出
    # Wireless 模式下 zenoh 未启动，需显式声明
    app = DoubaoVoiceApp(running_on_wireless=True)
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        logger.info("用户中断")
        app.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
