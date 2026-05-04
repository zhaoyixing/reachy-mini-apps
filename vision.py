"""视觉分析模块 —— 支持豆包方舟 或 Gemini 做画面描述。

豆包方舟（推荐国内使用）：
  1. 去 https://console.volcengine.com/ark/ 开通方舟大模型平台
  2. 创建 API Key（和语音 API Key 不同）
  3. 创建视觉模型推理接入点（如 doubao-1.5-vision-pro-32k）
  4. 把 ARK_API_KEY 和接入点 Endpoint ID 填入 .env

Gemini（免申请、免费）：
  申请地址：https://aistudio.google.com/app/apikey
  免费额度 1500 请求/天
"""
from __future__ import annotations
import base64
import io
import logging
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


def _encode_frame(frame_bgr: np.ndarray, quality: int = 85) -> str:
    """BGR numpy 数组 → base64 JPEG。"""
    rgb = frame_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _analyze_doubao_ark(
    frame_bgr: np.ndarray,
    ark_api_key: str,
    endpoint_id: str,
    prompt: str,
    timeout: float,
) -> Optional[str]:
    """调用豆包方舟视觉模型（OpenAI-compatible 格式）。"""
    b64 = _encode_frame(frame_bgr)
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Authorization": f"Bearer {ark_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": endpoint_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 256,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "").strip()
            if text:
                logger.info("[Vision-豆包] %s", text[:120])
                return text
    except requests.exceptions.Timeout:
        logger.warning("Vision analysis timeout (doubao)")
    except Exception as exc:
        logger.warning("Vision analysis failed (doubao): %s", exc)
    return None


def _analyze_gemini(
    frame_bgr: np.ndarray,
    api_key: str,
    prompt: str,
    timeout: float,
) -> Optional[str]:
    """调用 Gemini 2.0 Flash 视觉模型。"""
    b64 = _encode_frame(frame_bgr)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]
        }]
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p.get("text", "") for p in parts if "text" in p]
            result = "".join(texts).strip()
            if result:
                logger.info("[Vision-Gemini] %s", result[:120])
                return result
    except requests.exceptions.Timeout:
        logger.warning("Vision analysis timeout (gemini)")
    except Exception as exc:
        logger.warning("Vision analysis failed (gemini): %s", exc)
    return None


def analyze_frame(
    frame_bgr: np.ndarray,
    prompt: str = "用一句话描述这张图片里的主要物体和场景。",
    timeout: float = 5.0,
    *,
    doubao_ark_key: Optional[str] = None,
    doubao_vision_endpoint: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
) -> Optional[str]:
    """分析单帧画面，优先使用豆包方舟，其次 Gemini。

    Args:
        frame_bgr: OpenCV BGR 格式 numpy 数组 (H, W, 3)
        prompt: 给模型的提示词
        timeout: 请求超时秒数
        doubao_ark_key: 豆包方舟 API Key
        doubao_vision_endpoint: 豆包方舟视觉模型接入点 ID（如 ep-2024xxxxx）
        gemini_api_key: Gemini API Key

    Returns:
        模型返回的描述文本，失败返回 None
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # 优先豆包方舟
    if doubao_ark_key and doubao_vision_endpoint:
        return _analyze_doubao_ark(
            frame_bgr, doubao_ark_key, doubao_vision_endpoint, prompt, timeout
        )

    # 其次 Gemini
    if gemini_api_key:
        return _analyze_gemini(frame_bgr, gemini_api_key, prompt, timeout)

    logger.debug("No vision backend configured")
    return None
