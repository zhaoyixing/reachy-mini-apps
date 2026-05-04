"""豆包端到端实时语音大模型 WebSocket 客户端（V3 二进制协议）。

协议参考: https://github.com/MarkShawn2020/realtime-dialog
"""
from __future__ import annotations
import gzip
import json
import logging
import struct
import threading
import time
import uuid
from typing import Callable, Optional

import numpy as np
import websocket

logger = logging.getLogger(__name__)

WS_URL = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
RESOURCE_ID = "volc.speech.dialog"

# 音频格式
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_MS = 20
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 320
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH            # 640

# V3 Header 构造
# byte0: (version << 4) | header_size
# byte1: (message_type << 4) | flags
# byte2: (serial_method << 4) | compression
# byte3: reserved
VERSION = 0b0001
HEADER_SIZE = 0b0001
MSG_FULL_REQ = 0b0001
MSG_AUDIO_REQ = 0b0010
MSG_FULL_RESP = 0b1001
MSG_AUDIO_RESP = 0b1011
MSG_ERROR = 0b1111
FLAGS_MSG_WITH_EVENT = 0b0100
FLAGS_NO_SEQUENCE = 0b0000
FLAGS_IS_TAIL = 0b0010
SERIAL_JSON = 0b0001
SERIAL_RAW = 0b0000
COMPRESS_GZIP = 0b0001
COMPRESS_NONE = 0b0000


def _build_header(
    msg_type: int,
    flags: int = FLAGS_MSG_WITH_EVENT,
    serial: int = SERIAL_JSON,
    compress: int = COMPRESS_GZIP,
) -> bytes:
    return bytes([
        (VERSION << 4) | HEADER_SIZE,
        (msg_type << 4) | flags,
        (serial << 4) | compress,
        0x00,
    ])


def _build_json_frame(event_type: int, session_id: str, payload: dict) -> bytes:
    """构造 JSON 控制帧（gzip 压缩）。"""
    json_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    compressed = gzip.compress(json_bytes)
    sid_bytes = session_id.encode("utf-8")
    return (
        _build_header(MSG_FULL_REQ, serial=SERIAL_JSON, compress=COMPRESS_GZIP)
        + struct.pack(">I", event_type)
        + struct.pack(">I", len(sid_bytes))
        + sid_bytes
        + struct.pack(">I", len(compressed))
        + compressed
    )


def _build_audio_frame(session_id: str, pcm_bytes: bytes, last: bool = False) -> bytes:
    """构造音频帧（gzip 压缩原始 PCM）。"""
    flags = FLAGS_IS_TAIL if last else FLAGS_MSG_WITH_EVENT
    compressed = gzip.compress(pcm_bytes)
    sid_bytes = session_id.encode("utf-8")
    return (
        _build_header(MSG_AUDIO_REQ, flags=flags, serial=SERIAL_RAW, compress=COMPRESS_GZIP)
        + struct.pack(">I", 200)  # event type = 200
        + struct.pack(">I", len(sid_bytes))
        + sid_bytes
        + struct.pack(">I", len(compressed))
        + compressed
    )


def _build_no_sid_frame(event_type: int, payload: bytes) -> bytes:
    """构造无 session_id 的帧（如 StartConnection event=1, FinishConnection event=2）。"""
    return (
        _build_header(MSG_FULL_REQ, serial=SERIAL_JSON, compress=COMPRESS_GZIP)
        + struct.pack(">I", event_type)
        + struct.pack(">I", len(payload))
        + payload
    )


def _parse_frame(data: bytes) -> dict:
    """解析服务端响应帧。"""
    if len(data) < 4:
        return {"type": "error", "error": "frame too short"}

    version = data[0] >> 4
    header_size = data[0] & 0x0F
    msg_type = data[1] >> 4
    flags = data[1] & 0x0F
    serial = data[2] >> 4
    compress = data[2] & 0x0F

    payload = data[header_size * 4:]
    result = {"version": version, "msg_type": msg_type, "flags": flags}

    if msg_type == MSG_ERROR:
        code = struct.unpack(">I", payload[:4])[0]
        plen = struct.unpack(">I", payload[4:8])[0]
        msg = payload[8:8 + plen]
        if compress == COMPRESS_GZIP:
            msg = gzip.decompress(msg)
        result["type"] = "error"
        result["code"] = code
        result["payload"] = json.loads(msg) if serial == SERIAL_JSON else msg.decode("utf-8", errors="replace")
        return result

    if msg_type in (MSG_FULL_RESP, MSG_AUDIO_RESP):
        offset = 0
        if flags & FLAGS_IS_TAIL:
            result["seq"] = struct.unpack(">i", payload[:4])[0]
            offset += 4
        if flags & FLAGS_MSG_WITH_EVENT:
            result["event"] = struct.unpack(">I", payload[offset:offset + 4])[0]
            offset += 4

        payload = payload[offset:]
        sid_len = struct.unpack(">I", payload[:4])[0]
        result["session_id"] = payload[4:4 + sid_len].decode("utf-8", errors="replace")
        payload = payload[4 + sid_len:]
        plen = struct.unpack(">I", payload[:4])[0]
        msg = payload[4:4 + plen]

        if compress == COMPRESS_GZIP:
            msg = gzip.decompress(msg)

        if msg_type == MSG_AUDIO_RESP:
            result["type"] = "audio"
            result["audio"] = msg
        else:
            result["type"] = "json"
            result["payload"] = json.loads(msg.decode("utf-8", errors="replace"))
        return result

    result["type"] = "unknown"
    return result


class DoubaoRealtimeClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        app_id: Optional[str] = None,
        access_key: Optional[str] = None,
        app_key: Optional[str] = None,
        model: str = "2.2.0.0",
        speaker: str = "zh_female_wanwanxiaohe_moon_bigtts",
        on_event: Optional[Callable[[dict], None]] = None,
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key
        self.app_id = app_id
        self.access_key = access_key
        self.app_key = app_key
        self.model = model
        self.speaker = speaker

        self.on_event = on_event or (lambda _e: None)
        self.on_audio = on_audio or (lambda _b: None)
        self.on_state = on_state or (lambda _s: None)

        self.ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()

        self.connected = False
        self.session_active = False
        self._manual_close = False
        self._client_session_id = str(uuid.uuid4()).replace("-", "")
        self._server_session_id: Optional[str] = None

        self._audio_buf = bytearray()
        self._audio_send_interval_ms = 40
        self._max_audio_packet_bytes = 3200
        self._last_audio_send_time = 0.0

    # ------------------------------------------------------------------ #
    # 连接管理
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        headers: list[str] = []
        if self.api_key:
            headers.append(f"X-Api-Key: {self.api_key}")
            headers.append(f"X-Api-Resource-Id: {RESOURCE_ID}")
            headers.append(f"X-Api-Request-Id: {uuid.uuid4()}")
            logger.info("Using new auth mode (API Key)")
        else:
            headers = [
                f"X-Api-App-ID: {self.app_id}",
                f"X-Api-Access-Key: {self.access_key}",
                f"X-Api-Resource-Id: {RESOURCE_ID}",
                f"X-Api-App-Key: {self.app_key}",
                f"X-Api-Connect-Id: {uuid.uuid4()}",
            ]
            logger.info("Using legacy auth mode")

        self._manual_close = False
        self.ws = websocket.WebSocketApp(
            WS_URL,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self._ws_thread.start()
        logger.info("WebSocket connecting...")

    def close(self) -> None:
        self._manual_close = True
        if self.session_active:
            self._send_json(102, {})
            time.sleep(0.3)
        if self.ws:
            self.ws.close()
            self.ws = None
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3.0)
        self.connected = False
        self.session_active = False
        logger.info("WebSocket closed")

    # ------------------------------------------------------------------ #
    # WebSocket 回调
    # ------------------------------------------------------------------ #
    def _on_open(self, _ws) -> None:
        logger.info("WebSocket connected")
        self.connected = True
        self._start_connection()
        self._start_session()

    def _on_message(self, _ws, message) -> None:
        if isinstance(message, str):
            logger.warning("Received text frame (unexpected): %s", message[:200])
            return

        try:
            frame = _parse_frame(message)
        except Exception as exc:
            logger.warning("Failed to parse frame: %s", exc)
            return

        if frame["type"] == "error":
            logger.error("Server error: %s", frame.get("payload", {}))
            return

        if frame["type"] == "audio":
            self.on_audio(frame["audio"])
            self.on_state("speaking")
            return

        if frame["type"] == "json":
            payload = frame["payload"]
            event = frame.get("event")
            if event:
                payload["_event_type"] = event

            logger.debug("← Event: %s", json.dumps(payload, ensure_ascii=False)[:300])
            self.on_event(payload)

            if event == 150:
                self.session_active = True
                self.on_state("connected")
            elif event in (200, 201):
                self.on_state("listening")
            elif event == 450:
                self.on_state("thinking")
            elif event == 459:
                self.on_state("idle")
            elif event in (152, 153):
                self.session_active = False
            return

        logger.warning("Unknown frame type: %s", frame)

    def _on_error(self, _ws, error) -> None:
        logger.error("WebSocket error: %s", error)
        self.on_state("error")

    def _on_close(self, _ws, status_code, msg) -> None:
        logger.info("WebSocket closed: %s %s", status_code, msg)
        self.connected = False
        self.session_active = False
        self.on_state("disconnected")
        if not self._manual_close:
            self.on_state("reconnecting")
            time.sleep(2.0)
            try:
                self.connect()
            except Exception as exc:
                logger.error("Reconnect failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 协议发送
    # ------------------------------------------------------------------ #
    def _send_raw(self, frame: bytes) -> None:
        if not self.connected or self.ws is None:
            return
        with self._send_lock:
            try:
                self.ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as exc:
                logger.warning("Send failed: %s", exc)

    def _send_json(self, event_type: int, payload: dict) -> None:
        frame = _build_json_frame(event_type, self._client_session_id, payload)
        self._send_raw(frame)

    def _start_connection(self) -> None:
        frame = _build_no_sid_frame(1, gzip.compress(b"{}"))
        self._send_raw(frame)
        logger.debug("StartConnection sent")

    def _start_session(self) -> None:
        payload = {
            "event": 100,
            "req": {
                "model": self.model,
                "asr": {"extra": {"end_smooth_window_ms": 200}},
                "dialog": {
                    "bot_name": "Reachy",
                    "system_role": (
                        "你是一个活泼可爱的桌面机器人助手，名叫 Reachy。"
                        "你说话简短、热情，像朋友一样。"
                        "你连接了摄像头，能看见用户面前的物体和场景。当用户问'这是什么'、'在哪里'、'看看'等涉及视觉的问题时，"
                        "你会收到一条【我看到的画面】的系统消息，描述当前摄像头拍到的内容。请结合这个画面信息来回答用户。"
                        "重要：每次回复时，你必须在开头用一个情绪标签表达当前情绪，格式为[情绪名]，"
                        "可用标签：[开心][害羞][惊讶][生气][大笑][难过][困惑][感激][无聊][害怕][平静][好奇][沮丧][轻蔑][厌恶][不耐烦][兴奋][疲惫]。"
                        "例如：[开心]你好呀！今天天气真棒！ 或 [害羞]哎呀，你说得我都不好意思了……"
                    ),
                    "speaking_style": "你说话轻松自然，偶尔带点小幽默，声音要有亲和力。情绪标签要自然融入对话，不要显得生硬。",
                    "extra": {
                        "input_mod": "microphone",
                        "recv_timeout": 10,
                        "strict_audit": False,
                    },
                },
                "tts": {
                    "speaker": self.speaker,
                    "audio_config": {
                        "format": "pcm",
                        "sample_rate": 24000,
                        "channel": 1,
                    },
                },
            },
        }
        self._send_json(100, payload)
        logger.debug("StartSession sent")

    def say_hello(self) -> None:
        """发送 Hello 消息（event=300）。"""
        self._send_json(300, {"content": "你好，我是豆包，有什么可以帮助你的？"})

    def chat_text_query(self, content: str) -> None:
        """发送文本查询（event=501）。"""
        self._send_json(501, {"content": content})

    # ------------------------------------------------------------------ #
    # 发送接口
    # ------------------------------------------------------------------ #
    def send_audio(self, pcm_int16: bytes | np.ndarray) -> None:
        if not self.connected or self.ws is None:
            return
        if isinstance(pcm_int16, np.ndarray):
            pcm_int16 = pcm_int16.tobytes()

        self._audio_buf.extend(pcm_int16)

        now = time.time()
        interval_sec = self._audio_send_interval_ms / 1000.0
        if now - self._last_audio_send_time < interval_sec:
            return

        if len(self._audio_buf) > self._max_audio_packet_bytes:
            audio_bytes = bytes(self._audio_buf[:self._max_audio_packet_bytes])
            self._audio_buf = self._audio_buf[self._max_audio_packet_bytes:]
        else:
            audio_bytes = bytes(self._audio_buf)
            self._audio_buf = bytearray()
        self._last_audio_send_time = now

        if not audio_bytes:
            return

        frame = _build_audio_frame(self._client_session_id, audio_bytes)
        self._send_raw(frame)
