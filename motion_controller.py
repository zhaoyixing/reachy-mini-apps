"""Reachy Mini 动作状态机——根据对话状态驱动机器人表情与动作。"""
from __future__ import annotations
import time
import logging
import threading
from typing import Tuple, Optional

import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves

from speech_tapper import SwayRollRT

logger = logging.getLogger(__name__)
CONTROL_HZ = 30.0


def to_homogeneous(pose) -> np.ndarray:
    """确保 pose 是 4x4 float64 矩阵。"""
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose, got {p.shape}")
    return p


class MotionController:
    """
    状态机：idle → listening → thinking → speaking → idle
    每个状态对应不同的头部/天线/身体动作。
    新增：支持 emotions library 中的情绪动作，在 speaking 时根据豆包情绪播放。
    """

    def __init__(self, robot: ReachyMini) -> None:
        self.robot = robot
        self._state: str = "idle"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # 语音响应算法
        self._sway = SwayRollRT()
        self._sway_offsets: dict = {
            "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0,
            "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
        }
        self._sway_lock = threading.Lock()

        # 监听时记住的声源方向
        self._last_doa_yaw_deg: float = 0.0

        # 情绪动作库
        try:
            self._emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            logger.info("Loaded emotions library: %s", self._emotions.list_moves())
        except Exception as exc:
            logger.warning("Failed to load emotions library: %s", exc)
            self._emotions = None

        self._emotion_map = {
            "开心": "cheerful1",
            "害羞": "downcast1",
            "惊讶": "amazed1",
            "生气": "furious1",
            "大笑": "laughing1",
            "难过": "downcast1",
            "困惑": "confused1",
            "感激": "grateful1",
            "无聊": "boredom1",
            "害怕": "fear1",
            "平静": "calming1",
            "好奇": "curious1",
            "沮丧": "frustrated1",
            "轻蔑": "contempt1",
            "厌恶": "disgusted1",
            "不耐烦": "impatient1",
            "兴奋": "enthusiastic1",
            "疲惫": "exhausted1",
        }

        self._current_emotion: Optional[RecordedMove] = None
        self._emotion_start_time: float = 0.0
        self._emotion_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def set_state(self, state: str) -> None:
        with self._lock:
            if self._state != state:
                logger.info("Motion state: %s → %s", self._state, state)
                self._state = state
                # 每次进入 speaking 且没有待播情绪时，自动选一个随机情绪兜底
                if state == "speaking":
                    with self._emotion_lock:
                        has_pending = self._current_emotion is not None
                    if not has_pending:
                        self._auto_trigger_random_emotion()

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def _auto_trigger_random_emotion(self) -> None:
        """如果没有精准情绪，随机挑一个情绪动作兜底，让机器人每次开口都生动。"""
        if self._emotions is None:
            return
        try:
            candidates = [v for v in self._emotion_map.values() if v]
            if not candidates:
                return
            import random
            move_name = random.choice(candidates)
            move = self._emotions.get(move_name)
            with self._emotion_lock:
                self._current_emotion = move
                self._emotion_start_time = time.monotonic()
            logger.info("Auto emotion: %s (%.2fs)", move_name, move.duration)
        except Exception as exc:
            logger.debug("Auto emotion failed: %s", exc)

    def trigger_emotion(self, name: str) -> None:
        """根据情绪名称触发对应的预录动作，在下次 speaking 时播放。"""
        if self._emotions is None:
            return
        move_name = self._emotion_map.get(name)
        if not move_name:
            logger.debug("Unknown emotion: %s", name)
            return
        try:
            move = self._emotions.get(move_name)
            with self._emotion_lock:
                self._current_emotion = move
                self._emotion_start_time = time.monotonic()
            logger.info("Emotion triggered: %s → %s (%.2fs)", name, move_name, move.duration)
        except Exception as exc:
            logger.warning("Failed to trigger emotion %s: %s", move_name, exc)

    def feed_speech_audio(self, pcm_int16: np.ndarray, sr: int = 16000) -> None:
        """把机器人正在播放的音频喂给 sway，实现说话时的头部晃动。"""
        results = self._sway.feed(pcm_int16, sr=sr)
        if results:
            with self._sway_lock:
                self._sway_offsets = results[-1]

    def reset_sway(self) -> None:
        self._sway.reset()
        with self._sway_lock:
            self._sway_offsets = {
                "x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0,
                "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
            }

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("MotionController started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("MotionController stopped")

    # ------------------------------------------------------------------ #
    # 控制循环
    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        dt = 1.0 / CONTROL_HZ
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            state = self.get_state()

            try:
                if state == "idle":
                    self._do_idle(t0)
                elif state == "listening":
                    self._do_listening(t0)
                elif state == "thinking":
                    self._do_thinking(t0)
                elif state == "speaking":
                    self._do_speaking(t0)
                else:
                    self._do_idle(t0)
            except Exception as exc:
                logger.warning("Motion loop error: %s", exc)

            elapsed = time.monotonic() - t0
            sleep_needed = dt - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    # ------------------------------------------------------------------ #
    # 各状态动作
    # ------------------------------------------------------------------ #
    def _do_idle(self, t: float) -> None:
        """缓慢呼吸 + 天线轻微反向摆动。"""
        z_mm = 5.0 * np.sin(2 * np.pi * 0.1 * t)
        ant_deg = 10.0 * np.sin(2 * np.pi * 0.5 * t)
        head = create_head_pose(0, 0, z_mm, 0, 0, 0, degrees=True, mm=True)
        self._set_target(head, (np.radians(ant_deg), np.radians(-ant_deg)), 0.0)

    def _do_listening(self, t: float) -> None:
        """安静倾听：只转向声源，其余保持静止。"""
        doa = None
        try:
            doa = self.robot.media.get_DoA()
        except Exception:
            pass

        yaw_deg = self._last_doa_yaw_deg
        if doa is not None:
            angle, speech_detected = doa
            if speech_detected:
                yaw_deg = np.degrees(angle)
                self._last_doa_yaw_deg = yaw_deg

        # 倾听时完全静止，不摇晃、不摆天线
        head = create_head_pose(0, 0, 0, 0, 0, yaw_deg, degrees=True, mm=True)
        self._set_target(head, (0.0, 0.0), 0.0)

    def _do_thinking(self, t: float) -> None:
        """思考时保持静止，不摇晃。"""
        head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True, mm=True)
        self._set_target(head, (0.0, 0.0), 0.0)

    def _do_speaking(self, _t: float) -> None:
        """优先播放情绪动作，其次语音驱动的头部晃动（Sway）+ 轻微天线活动。"""
        # 优先播放情绪动作
        with self._emotion_lock:
            emotion = self._current_emotion
            emotion_start = self._emotion_start_time

        if emotion is not None:
            elapsed = time.monotonic() - emotion_start
            if elapsed < emotion.duration:
                try:
                    head, antennas, body_yaw = emotion.evaluate(elapsed)
                    self._set_target(head, antennas, body_yaw)
                except Exception as exc:
                    logger.debug("Emotion evaluate failed: %s", exc)
                return
            else:
                with self._emotion_lock:
                    self._current_emotion = None

        # 原有 sway 逻辑
        with self._sway_lock:
            off = dict(self._sway_offsets)

        head = create_head_pose(
            off["x_mm"], off["y_mm"], off["z_mm"],
            off["roll_deg"], off["pitch_deg"], off["yaw_deg"],
            degrees=True, mm=True,
        )
        # 天线做轻微伴音摆动
        ant = 0.1 * np.sin(2 * np.pi * 3.0 * _t)
        self._set_target(head, (ant, -ant), 0.0)

    # ------------------------------------------------------------------ #
    # 底层控制
    # ------------------------------------------------------------------ #
    def _set_target(
        self,
        head_pose,
        antennas: Tuple[float, float],
        body_yaw: float,
    ) -> None:
        try:
            self.robot.set_target(head_pose, antennas, body_yaw)
        except Exception as exc:
            # 控制指令失败时静默忽略，避免日志刷屏
            logger.debug("set_target failed: %s", exc)
