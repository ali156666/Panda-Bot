from __future__ import annotations

import base64
import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_OUTPUT_DIR = str((Path(__file__).resolve().parent / "outputs").resolve())
TTS_DEFAULT_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_DEFAULT_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_DEFAULT_FORMAT = "opus"
TTS_DEFAULT_OUTPUT_DIR = str((Path(__file__).resolve().parent / "outputs" / "tts").resolve())

MIME_EXT_MAP = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Set it in the env or .env file.")
    return api_key


def _get_client():
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Missing dependency google-genai. Install requirements.txt.") from exc
    return genai.Client(api_key=_get_api_key())


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_image_paths(image_paths: Any) -> list[str]:
    if image_paths is None:
        return []
    if isinstance(image_paths, list):
        return [str(p) for p in image_paths if str(p).strip()]
    return [str(image_paths)]


def _load_images(image_paths: list[str]):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency Pillow. Install requirements.txt.") from exc

    images = []
    for path in image_paths:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(file_path) as img:
            images.append(img.copy())
    return images


def _extract_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None)
    if parts:
        return list(parts)
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    if content and getattr(content, "parts", None):
        return list(content.parts)
    return []


def _ext_from_mime(mime_type: str | None) -> str:
    if not mime_type:
        return ".png"
    mime_type = mime_type.lower()
    if mime_type in MIME_EXT_MAP:
        return MIME_EXT_MAP[mime_type]
    return mimetypes.guess_extension(mime_type) or ".png"


def _save_parts(parts: list[Any], output_dir: Path, prefix: str) -> list[str]:
    saved_paths: list[str] = []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue
        mime_type = getattr(inline_data, "mime_type", None)
        ext = _ext_from_mime(mime_type)
        filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
        file_path = output_dir / filename

        data = getattr(inline_data, "data", None)
        if data:
            if isinstance(data, str):
                raw = base64.b64decode(data)
            else:
                raw = data
            file_path.write_bytes(raw)
            saved_paths.append(str(file_path))
            continue

        if hasattr(part, "as_image"):
            image = part.as_image()
            image.save(file_path)
            saved_paths.append(str(file_path))
    return saved_paths


def _generate_images(
    *,
    prompt: str,
    image_paths: list[str] | None,
    output_dir: str,
    prefix: str,
) -> str:
    if not prompt or not str(prompt).strip():
        return "Error: prompt cannot be empty."

    try:
        normalized_paths = _normalize_image_paths(image_paths)
        if image_paths is not None and not normalized_paths:
            return "Error: image_paths cannot be empty."

        client = _get_client()
        contents: list[Any] = [prompt]
        if normalized_paths:
            images = _load_images(normalized_paths)
            contents.extend(images)

        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
        )
        parts = _extract_parts(response)
        output_path = _ensure_output_dir(output_dir)
        saved_paths = _save_parts(parts, output_path, prefix)
        if not saved_paths:
            return "No images returned."
        return "\n".join(saved_paths)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def gemini_text_to_image(prompt: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """Text-to-image. Returns local file paths."""
    return _generate_images(
        prompt=prompt,
        image_paths=None,
        output_dir=output_dir,
        prefix="gemini_text_to_image",
    )


@tool
def gemini_image_edit(
    prompt: str,
    image_paths: list[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Edit images with text + image inputs. Returns local file paths."""
    return _generate_images(
        prompt=prompt,
        image_paths=image_paths,
        output_dir=output_dir,
        prefix="gemini_image_edit",
    )


@tool
def gemini_image_to_image(
    prompt: str,
    image_paths: list[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Image-to-image variations with text + image inputs. Returns local file paths."""
    return _generate_images(
        prompt=prompt,
        image_paths=image_paths,
        output_dir=output_dir,
        prefix="gemini_image_to_image",
    )


def get_gemini_image_tools() -> list[Any]:
    return [gemini_text_to_image, gemini_image_edit, gemini_image_to_image]


def _get_tts_api_key() -> str:
    api_key = (
        os.getenv("TTS_KEY")
        or os.getenv("TTS_API_KEY")
        or os.getenv("MEMORY_SUMMARY_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing TTS key. Set TTS_KEY or TTS_API_KEY (fallback: MEMORY_SUMMARY_KEY / OPENAI_API_KEY)."
        )
    return api_key


def _get_tts_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency openai. Install requirements.txt or pip install openai.") from exc
    base_url = os.getenv("TTS_BASE_URL") or os.getenv("TTS_URL") or os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": _get_tts_api_key()}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _tts_output_path(output_dir: str, response_format: str) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    ext = f".{response_format.lower().strip()}" if response_format else ".mp3"
    return (path / f"tts_{uuid.uuid4().hex}{ext}").resolve()


def _normalize_speed(speed: float | None) -> float | None:
    if speed is None:
        return None
    value = float(speed)
    if value < 0.25 or value > 4.0:
        raise ValueError("speed 取值范围为 0.25~4.0")
    return value


@tool
def tts_generate_audio(
    text: str,
    voice: str = "",
    model: str = "",
    speed: float | None = None,
    instructions: str = "",
) -> str:
    """生成语音文件（固定输出 opus），返回 JSON 字符串：包含绝对路径与免责声明提示。"""
    if not text or not str(text).strip():
        return "Error: text 不能为空。"

    # 统一使用默认格式与默认输出目录，避免模型传入不符合预期的参数。
    voice_name = (voice or TTS_DEFAULT_VOICE).strip()
    model_name = (model or TTS_DEFAULT_MODEL).strip()
    fmt = TTS_DEFAULT_FORMAT.strip().lower()
    out_dir = TTS_DEFAULT_OUTPUT_DIR

    try:
        speed_value = _normalize_speed(speed)
    except Exception as exc:
        return f"Error: {exc}"

    try:
        client = _get_tts_client()
    except Exception as exc:
        return f"Error: {exc}"

    output_path = _tts_output_path(out_dir, fmt)
    request_kwargs = {
        "model": model_name,
        "voice": voice_name,
        "input": str(text).strip(),
        "response_format": fmt,
    }
    if speed_value is not None:
        request_kwargs["speed"] = speed_value
    if instructions and model_name not in {"tts-1", "tts-1-hd"}:
        request_kwargs["instructions"] = str(instructions).strip()

    try:
        with client.audio.speech.with_streaming_response.create(**request_kwargs) as response:
            response.stream_to_file(output_path)
    except Exception as exc:
        return f"Error: 生成语音失败：{exc}"

    result = {
        "audio_path": str(output_path.resolve()),
        "format": fmt,
        "disclaimer": "请告知用户这是 AI 生成的语音，而非真人。",
    }
    return json.dumps(result, ensure_ascii=False)


def get_tts_tools() -> list[Any]:
    return [tts_generate_audio]
