import asyncio
import atexit
import contextvars
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO


LOG_FILE_PATH = Path(os.getenv("FEISHU_BOT_LOG", "feishu_bot.log")).resolve()
_LOG_FILE_STREAM: Optional[TextIO] = None


class _TeeStream:
    """Mirror writes to both original stream and a log file."""

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror
        self.encoding = getattr(primary, "encoding", "utf-8")
        self.errors = getattr(primary, "errors", "replace")

    def write(self, data: str) -> int:
        written = self._primary.write(data)
        self._mirror.write(data)
        return written

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        isatty_fn = getattr(self._primary, "isatty", None)
        return bool(isatty_fn()) if callable(isatty_fn) else False

    def fileno(self) -> int:
        fileno_fn = getattr(self._primary, "fileno", None)
        if callable(fileno_fn):
            return int(fileno_fn())
        raise OSError("fileno is not supported")

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


def _close_log_stream() -> None:
    global _LOG_FILE_STREAM
    if isinstance(sys.stdout, _TeeStream):
        sys.stdout = sys.stdout._primary
    if isinstance(sys.stderr, _TeeStream):
        sys.stderr = sys.stderr._primary
    stream = _LOG_FILE_STREAM
    _LOG_FILE_STREAM = None
    if stream is None:
        return
    try:
        stream.flush()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass


def _install_stdio_tee(path: Path) -> None:
    """Capture stdout/stderr to a persistent log file."""
    global _LOG_FILE_STREAM
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE_STREAM = path.open("a", encoding="utf-8", errors="replace", buffering=1)
        if not isinstance(sys.stdout, _TeeStream):
            sys.stdout = _TeeStream(sys.stdout, _LOG_FILE_STREAM)
        if not isinstance(sys.stderr, _TeeStream):
            sys.stderr = _TeeStream(sys.stderr, _LOG_FILE_STREAM)
    except Exception as exc:
        print(f"[log] failed to initialize log file {path}: {exc}")


_install_stdio_tee(LOG_FILE_PATH)
atexit.register(_close_log_stream)

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateFileRequest,
    CreateFileRequestBody,
    CreateImageRequest,
    CreateImageRequestBody,
    GetMessageResourceRequest,
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
)

from langchain_core.tools import tool

import deepagent_demo as runtime
from deepagent.session.manager import SessionManager
from scheduler import (
    CronService,
    HeartbeatConfig,
    HeartbeatRunner,
    HeartbeatWake,
    SchedulerStore,
    run_heartbeat_once,
)
from scheduler.cron import CronDeps
from scheduler.heartbeat import HeartbeatDeps
from scheduler.utils import now_ms


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing environment variable: {name}")
        sys.exit(1)
    return value


def _setup_lark_env() -> tuple[str, str]:
    app_id = _require_env("APP_ID")
    app_secret = _require_env("APP_SECRET")
    if hasattr(lark, "APP_ID"):
        lark.APP_ID = app_id
    if hasattr(lark, "APP_SECRET"):
        lark.APP_SECRET = app_secret
    base_domain = os.getenv("BASE_DOMAIN")
    if base_domain and hasattr(lark, "BASE_DOMAIN"):
        lark.BASE_DOMAIN = base_domain
    # 缁備胶鏁ゆ禒锝囨倞閻滎垰顣ㄩ崣姗€鍣洪敍宀勪缉閸?SDK 鐠ч绗夐崣顖滄暏娴狅絿鎮婇妴?    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)
    return app_id, app_secret


def _truncate_text(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ... (truncated)"


def _safe_getattr(obj, path: str, default: str = "") -> str:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return default
    return str(current)


@dataclass
class _FeishuContext:
    client: lark.Client
    chat_id: str
    chat_type: str
    sender_open_id: str = ""
    sender_user_id: str = ""


_FEISHU_CONTEXT: contextvars.ContextVar[Optional[_FeishuContext]] = contextvars.ContextVar(
    "feishu_context",
    default=None,
)

if os.name == "nt":
    _WINDOWS_RUNTIME_ROOT = Path(
        os.getenv(
            "DEEPAGENT_RUNTIME_ROOT",
            str(
                Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
                / "deepagent_runtime"
            ),
        )
    ).resolve()
    _DEFAULT_MEDIA_OUTPUT_DIR = str((Path.cwd() / "outputs" / "feishu_media").resolve())
    _DEFAULT_INBOUND_FILE_DIR = str((Path.cwd() / "outputs" / "feishu_inbound").resolve())
    _DEFAULT_SCHEDULER_DB_PATH = str((_WINDOWS_RUNTIME_ROOT / "scheduler.sqlite").resolve())
    _DEFAULT_MEMORY_ROOT = str((_WINDOWS_RUNTIME_ROOT / "memory" / "feishu").resolve())
    _DEFAULT_MEMORY_DB_PATH = str(
        (_WINDOWS_RUNTIME_ROOT / "memory" / "feishu" / "memory.sqlite").resolve()
    )
else:
    _DEFAULT_MEDIA_OUTPUT_DIR = "/opt/deepagent/outputs/feishu_media"
    _DEFAULT_INBOUND_FILE_DIR = "/opt/deepagent/outputs/feishu_inbound"
    _DEFAULT_SCHEDULER_DB_PATH = "/opt/deepagent/scheduler.sqlite"
    _DEFAULT_MEMORY_ROOT = "/opt/deepagent/memory/feishu"
    _DEFAULT_MEMORY_DB_PATH = "/opt/deepagent/memory/feishu/memory.sqlite"

MEDIA_OUTPUT_DIR = Path(os.getenv("MEDIA_OUTPUT_DIR", _DEFAULT_MEDIA_OUTPUT_DIR)).resolve()
INBOUND_FILE_DIR = Path(os.getenv("FEISHU_INBOUND_DIR", _DEFAULT_INBOUND_FILE_DIR)).resolve()

_SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".bmp",
    ".ico",
}

_FILE_TYPE_MAP = {
    ".doc": "doc",
    ".docx": "doc",
    ".ppt": "ppt",
    ".pptx": "ppt",
    ".xls": "xls",
    ".xlsx": "xls",
    ".csv": "stream",
}

FEISHU_MEDIA_TOOL_PROMPT = (
    "You can use browser tools for web actions when needed. "
    "When you need to send local image/audio/video/file to Feishu, call feishu_send_media "
    "with type=image/audio/video/file and an absolute local path. "
    "If you generate audio from text, clearly tell the user it is AI-generated. "
    "For scheduling, always use cron_add/cron_list/cron_remove/cron_run tools. "
    "Do not claim a schedule is created unless the cron tool call succeeded."
)

LOG_TAIL_CHARS = int(os.getenv("LOG_TAIL_CHARS", "500"))
LOG_SUMMARY_WORDS = int(os.getenv("LOG_SUMMARY_WORDS", "1000"))



def _read_log_tail(path: Path = LOG_FILE_PATH, chars: int = LOG_TAIL_CHARS) -> str:
    """Return the last `chars` characters from the bot log file."""
    try:
        if not path.is_file():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if chars <= 0:
            return ""
        return content[-chars:].strip()
    except Exception as exc:
        print(f"[log] failed to read log tail: {exc}")
        return ""


def _read_log_tail_words(path: Path = LOG_FILE_PATH, max_words: int = LOG_SUMMARY_WORDS) -> str:
    """Return the last `max_words` words from the bot log file."""
    try:
        if not path.is_file() or max_words <= 0:
            return ""
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return ""
        words = content.split()
        if not words:
            return ""
        return " ".join(words[-max_words:]).strip()
    except Exception as exc:
        print(f"[log] failed to read log words: {exc}")
        return ""



CHANNEL_FEISHU = "feishu"
DEFAULT_AGENT_ID = os.getenv("DEEPAGENT_AGENT_ID", "main")
SCHEDULER_DB_PATH = Path(
    os.getenv("SCHEDULER_DB_PATH", _DEFAULT_SCHEDULER_DB_PATH)
).resolve()
FEISHU_MEMORY_ROOT = Path(
    os.getenv("FEISHU_MEMORY_ROOT", _DEFAULT_MEMORY_ROOT)
).resolve()
FEISHU_MEMORY_DB_PATH = Path(
    os.getenv("FEISHU_MEMORY_DB_PATH", _DEFAULT_MEMORY_DB_PATH)
).resolve()
_CRON_SERVICE: Optional[CronService] = None


def _bind_cron_service(service: CronService) -> None:
    global _CRON_SERVICE
    _CRON_SERVICE = service


class MainLaneTracker:
    def __init__(self) -> None:
        self._count = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            self._count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self._lock:
            self._count = max(0, self._count - 1)

    def is_busy(self) -> bool:
        return self._count > 0


def _build_heartbeat_config() -> HeartbeatConfig:
    def _int_env(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    dedupe_hours = _int_env("HEARTBEAT_DEDUPE_HOURS", 24)
    return HeartbeatConfig(
        every=os.getenv("HEARTBEAT_EVERY", "30m"),
        active_start=os.getenv("HEARTBEAT_ACTIVE_START"),
        active_end=os.getenv("HEARTBEAT_ACTIVE_END"),
        active_timezone=os.getenv("HEARTBEAT_ACTIVE_TZ"),
        prompt=os.getenv("HEARTBEAT_PROMPT"),
        ack_max_chars=_int_env("HEARTBEAT_ACK_MAX_CHARS", 300),
        dedupe_window_ms=max(0, dedupe_hours) * 60 * 60 * 1000,
    )


def _resolve_heartbeat_workspace_root() -> Path:
    root = Path(runtime.memory_root) / "agents"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_feishu_memory_manager() -> runtime.MemoryManager | None:
    try:
        FEISHU_MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        return runtime.MemoryManager(
            memory_dir=FEISHU_MEMORY_ROOT,
            db_path=FEISHU_MEMORY_DB_PATH,
            summary_model=runtime.memory_summary_model,
            embeddings=runtime.memory_embeddings,
            max_messages=runtime.memory_max_messages,
            min_score=runtime.memory_min_score,
            summary_every=runtime.memory_summary_every,
            habits_root=FEISHU_MEMORY_ROOT / "habits",
        )
    except Exception as exc:
        print(f"[memory] 妞嬬偘鍔熺拋鏉跨箓閸掓繂顫愰崠鏍с亼鐠? {exc}")
        return None


def _enter_feishu_context(
    client: lark.Client,
    chat_id: str,
    chat_type: str,
    sender_open_id: str = "",
    sender_user_id: str = "",
):
    return _FEISHU_CONTEXT.set(
        _FeishuContext(
            client=client,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_open_id=sender_open_id,
            sender_user_id=sender_user_id,
        )
    )


def _exit_feishu_context(token) -> None:
    _FEISHU_CONTEXT.reset(token)




def _resolve_media_path(raw_path: str) -> Path:
    if not raw_path or not str(raw_path).strip():
        raise ValueError("path is required")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        raise ValueError("path must be absolute")
    path = path.resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return path



def _ensure_output_dir() -> Path:
    MEDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_OUTPUT_DIR


def _ensure_inbound_dir() -> Path:
    INBOUND_FILE_DIR.mkdir(parents=True, exist_ok=True)
    return INBOUND_FILE_DIR


def _parse_message_content(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content or "{}")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_local_filename(name: str, fallback: str) -> str:
    candidate = Path(str(name or fallback)).name.strip()
    if not candidate:
        candidate = fallback
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in candidate)
    cleaned = cleaned.strip().strip(".")
    return cleaned or fallback



def _download_message_resource(
    client: lark.Client,
    *,
    message_id: str,
    resource_key: str,
    resource_type: str,
    file_name_hint: str = "",
) -> tuple[Path, int, str]:
    request = (
        GetMessageResourceRequest.builder()
        .message_id(message_id)
        .file_key(resource_key)
        .type(resource_type)
        .build()
    )
    response = client.im.v1.message_resource.get(request)
    _ensure_success(response, "download message resource failed")

    file_obj = getattr(response, "file", None)
    if file_obj is None:
        raise RuntimeError("message resource has no file stream")

    raw = file_obj.read()
    if raw is None:
        raw = b""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    if not isinstance(raw, (bytes, bytearray)):
        raise RuntimeError("message resource returned an unsupported payload type")

    file_name_resp = str(getattr(response, "file_name", "") or "").strip()
    original_name = file_name_hint.strip() or file_name_resp or f"{resource_type}_{resource_key}"
    safe_name = _safe_local_filename(
        original_name,
        f"{resource_type}_{resource_key}.bin",
    )
    local_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    out_path = _ensure_inbound_dir() / local_name
    data = bytes(raw)
    out_path.write_bytes(data)
    return out_path, len(data), original_name



def _infer_image_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {ext or 'unknown'}")
    # 娑撳﹣绱堕崶鍓у閹恒儱褰涚憰浣圭湴 image_type 娑撹櫣鏁ら柅鏃傝閸ㄥ绱濋崣鎴︹偓浣圭Х閹垰婧€閺咁垯濞囬悽?message
    return "message"


def _infer_file_type(path: Path) -> str:
    return _FILE_TYPE_MAP.get(path.suffix.lower(), "stream")


def _ensure_opus(path: Path) -> Path:
    if path.suffix.lower() == ".opus":
        return path
    output_dir = _ensure_output_dir()
    output_path = output_dir / f"{path.stem}_{uuid.uuid4().hex}.opus"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-c:a",
        "libopus",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg transcode failed: {detail or 'unknown error'}")
    return output_path


def _ensure_success(response, action: str) -> None:
    if response.success():
        return
    raise RuntimeError(
        f"{action}婢惰精瑙﹂敍瀹憃de={response.code} msg={response.msg} log_id={response.get_log_id()}"
    )



def _send_message(
    client: lark.Client,
    receive_id_type: str,
    receive_id: str,
    msg_type: str,
    payload: dict,
) -> None:
    content = json.dumps(payload, ensure_ascii=False)
    request = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        .build()
    )
    response = client.im.v1.message.create(request)
    _ensure_success(response, "send message failed")



def _select_receive_target(ctx: _FeishuContext) -> tuple[str, str]:
    if ctx.sender_user_id:
        return "user_id", ctx.sender_user_id
    if ctx.sender_open_id:
        return "open_id", ctx.sender_open_id
    return "chat_id", ctx.chat_id



def _upload_image(client: lark.Client, path: Path) -> str:
    image_type = _infer_image_type(path)
    with path.open("rb") as f:
        request = (
            CreateImageRequest.builder()
            .request_body(
                CreateImageRequestBody.builder()
                .image_type(image_type)
                .image(f)
                .build()
            )
            .build()
        )
        response = client.im.v1.image.create(request)
    _ensure_success(response, "upload image failed")
    image_key = getattr(getattr(response, "data", None), "image_key", None)
    if not image_key:
        raise RuntimeError("upload image succeeded but image_key is empty")
    return image_key




def _upload_file(
    client: lark.Client,
    path: Path,
    file_type: str,
    file_name: str,
    duration: Optional[int] = None,
) -> str:
    with path.open("rb") as f:
        body_builder = (
            CreateFileRequestBody.builder()
            .file_type(file_type)
            .file_name(file_name)
            .file(f)
        )
        if duration is not None:
            body_builder.duration(duration)
        request = CreateFileRequest.builder().request_body(body_builder.build()).build()
        response = client.im.v1.file.create(request)
    _ensure_success(response, "upload file failed")
    file_key = getattr(getattr(response, "data", None), "file_key", None)
    if not file_key:
        raise RuntimeError("upload file succeeded but file_key is empty")
    return file_key




@tool
def feishu_send_media(
    type: str,
    path: str,
    text: str = "",
    file_name: str = "",
    duration: Optional[int] = None,
) -> str:
    """Send local media (image/audio/video/file) to the active Feishu peer chat."""
    ctx = _FEISHU_CONTEXT.get()
    if ctx is None:
        return "Error: Feishu context not initialized for this request."
    if ctx.chat_type != "p2p":
        return "Error: feishu_send_media currently supports only p2p chats."

    try:
        media_path = _resolve_media_path(path)
    except Exception as exc:
        return f"Error: {exc}"

    if text and str(text).strip():
        try:
            receive_type, receive_id = _select_receive_target(ctx)
            _send_message(
                ctx.client,
                receive_type,
                receive_id,
                "text",
                {"text": str(text).strip()},
            )
        except Exception as exc:
            return f"Error: failed to send preface text: {exc}"

    kind = (type or "").strip().lower()
    try:
        if kind == "image":
            image_key = _upload_image(ctx.client, media_path)
            receive_type, receive_id = _select_receive_target(ctx)
            _send_message(
                ctx.client,
                receive_type,
                receive_id,
                "image",
                {"image_key": image_key},
            )
            return "OK: image sent"

        if kind == "audio":
            opus_path = _ensure_opus(media_path)
            resolved_name = file_name.strip() if file_name else opus_path.name
            if not resolved_name.lower().endswith(".opus"):
                resolved_name = Path(resolved_name).with_suffix(".opus").name
            duration_value = None
            if duration is not None:
                duration_value = int(duration)
            file_key = _upload_file(
                ctx.client,
                opus_path,
                file_type="opus",
                file_name=resolved_name,
                duration=duration_value,
            )
            receive_type, receive_id = _select_receive_target(ctx)
            _send_message(
                ctx.client,
                receive_type,
                receive_id,
                "audio",
                {"file_key": file_key},
            )
            return "OK: audio sent"

        if kind == "video":
            resolved_name = file_name.strip() if file_name else media_path.name
            if not resolved_name.lower().endswith(".mp4"):
                resolved_name = Path(resolved_name).with_suffix(".mp4").name
            file_key = _upload_file(
                ctx.client,
                media_path,
                file_type="mp4",
                file_name=resolved_name,
            )
            receive_type, receive_id = _select_receive_target(ctx)
            _send_message(
                ctx.client,
                receive_type,
                receive_id,
                "media",
                {"file_key": file_key},
            )
            return "OK: video sent"

        if kind == "file":
            resolved_name = file_name.strip() if file_name else media_path.name
            file_type = _infer_file_type(media_path)
            file_key = _upload_file(
                ctx.client,
                media_path,
                file_type=file_type,
                file_name=resolved_name,
            )
            receive_type, receive_id = _select_receive_target(ctx)
            _send_message(
                ctx.client,
                receive_type,
                receive_id,
                "file",
                {"file_key": file_key},
            )
            return "OK: file sent"

        return "Error: unsupported type, expected image/audio/video/file."
    except Exception as exc:
        return f"Error: failed to send media: {exc}"


def _get_cron_service() -> tuple[Optional[CronService], Optional[str]]:
    service = _CRON_SERVICE
    if service is None:
        return None, "Error: cron service is not initialized yet."
    return service, None


def _format_cron_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload", {}) or {}
    state = job.get("state", {}) or {}
    payload_text = str(payload.get("text", ""))
    if len(payload_text) > 300:
        payload_text = payload_text[:300] + "..."
    return {
        "job_id": job.get("job_id"),
        "name": job.get("name"),
        "description": job.get("description"),
        "enabled": bool(job.get("enabled", True)),
        "delete_after_run": bool(job.get("delete_after_run", False)),
        "schedule": job.get("schedule", {}),
        "session_target": job.get("session_target"),
        "wake_mode": job.get("wake_mode"),
        "payload_kind": job.get("payload_kind"),
        "payload_text": payload_text,
        "next_run_at": state.get("next_run_at"),
        "last_run_at": state.get("last_run_at"),
        "last_status": state.get("last_status"),
        "last_error": state.get("last_error"),
        "updated_at": job.get("updated_at"),
    }


def _build_schedule(
    *,
    schedule_kind: str,
    cron_expr: str,
    timezone: str,
    every_ms: int,
    at_ms: int,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    kind = str(schedule_kind or "").strip().lower()
    if kind == "cron":
        expr = str(cron_expr or "").strip()
        if not expr:
            return None, "cron_expr is required when schedule_kind='cron'"
        schedule: dict[str, Any] = {"kind": "cron", "expr": expr}
        tz = str(timezone or "").strip()
        if tz:
            schedule["tz"] = tz
        return schedule, None
    if kind == "every":
        if int(every_ms) <= 0:
            return None, "every_ms must be > 0 when schedule_kind='every'"
        return {"kind": "every", "every_ms": int(every_ms)}, None
    if kind == "at":
        if int(at_ms) <= 0:
            return None, "at_ms must be unix timestamp in milliseconds when schedule_kind='at'"
        return {"kind": "at", "at_ms": int(at_ms)}, None
    return None, "schedule_kind must be one of: cron, every, at"


@tool
async def cron_status() -> str:
    """Get cron scheduler status, including job count and next wake time."""
    service, err = _get_cron_service()
    if err:
        return err
    status = await service.status()
    data = {
        "enabled": status.enabled,
        "store_path": status.store_path,
        "jobs": status.jobs,
        "next_wake_at_ms": status.next_wake_at_ms,
    }
    return json.dumps(data, ensure_ascii=False)


@tool
async def cron_list(include_disabled: bool = False, limit: int = 50) -> str:
    """List cron jobs. Use include_disabled=true to include removed/disabled jobs."""
    service, err = _get_cron_service()
    if err:
        return err
    safe_limit = max(1, min(int(limit), 200))
    jobs = await service.list(include_disabled=bool(include_disabled))
    items = [_format_cron_job(job) for job in jobs[:safe_limit]]
    return json.dumps(
        {"count": len(jobs), "returned": len(items), "jobs": items},
        ensure_ascii=False,
    )


@tool
async def cron_add(
    name: str,
    text: str,
    schedule_kind: str = "cron",
    cron_expr: str = "",
    timezone: str = "",
    every_ms: int = 0,
    at_ms: int = 0,
    wake_mode: str = "now",
    description: str = "",
    enabled: bool = True,
    delete_after_run: bool = False,
) -> str:
    """Create a cron job for main session system events.

    Examples:
    - daily 20:45: schedule_kind='cron', cron_expr='45 20 * * *', timezone='Asia/Shanghai'
    - every 30 minutes: schedule_kind='every', every_ms=1800000
    - one-shot timestamp: schedule_kind='at', at_ms=1739018700000
    """
    service, err = _get_cron_service()
    if err:
        return err

    text_value = str(text or "").strip()
    if not text_value:
        return "Error: text is required."

    wake_mode_value = str(wake_mode or "").strip().lower()
    if wake_mode_value not in {"now", "next-heartbeat"}:
        return "Error: wake_mode must be 'now' or 'next-heartbeat'."

    schedule, schedule_error = _build_schedule(
        schedule_kind=schedule_kind,
        cron_expr=cron_expr,
        timezone=timezone,
        every_ms=every_ms,
        at_ms=at_ms,
    )
    if schedule_error:
        return f"Error: {schedule_error}"

    job_input: dict[str, Any] = {
        "name": str(name or "").strip() or "Cron Job",
        "description": str(description or "").strip(),
        "enabled": bool(enabled),
        "delete_after_run": bool(delete_after_run),
        "schedule": schedule,
        "session_target": "main",
        "wake_mode": wake_mode_value,
        "payload": {"kind": "systemEvent", "text": text_value},
    }
    created = await service.add(job_input)
    return json.dumps({"ok": True, "job": _format_cron_job(created)}, ensure_ascii=False)


@tool
async def cron_remove(job_id: str) -> str:
    """Disable/remove a cron job by job_id."""
    service, err = _get_cron_service()
    if err:
        return err
    job_id_value = str(job_id or "").strip()
    if not job_id_value:
        return "Error: job_id is required."
    result = await service.remove(job_id_value)
    return json.dumps(
        {"ok": result.ok, "removed": result.removed, "job_id": job_id_value},
        ensure_ascii=False,
    )


@tool
async def cron_run(job_id: str, mode: str = "force") -> str:
    """Run a cron job immediately.

    mode='force' runs regardless of due time; mode='due' only runs when due.
    """
    service, err = _get_cron_service()
    if err:
        return err
    job_id_value = str(job_id or "").strip()
    if not job_id_value:
        return "Error: job_id is required."
    mode_value = str(mode or "").strip().lower()
    if mode_value not in {"force", "due"}:
        return "Error: mode must be 'force' or 'due'."
    result = await service.run(job_id_value, mode=mode_value)
    return json.dumps(
        {"ok": result.ok, "ran": result.ran, "reason": result.reason, "job_id": job_id_value},
        ensure_ascii=False,
    )



def _extract_text_message(data: P2ImMessageReceiveV1) -> Optional[str]:
    message = data.event.message
    if message.chat_type != "p2p":
        return None
    if message.message_type != "text":
        return None
    try:
        payload = json.loads(message.content or "{}")
    except json.JSONDecodeError:
        return None
    text = payload.get("text", "")
    return text.strip() if isinstance(text, str) else None


def _extract_file_like_message(data: P2ImMessageReceiveV1, client: lark.Client) -> Optional[str]:
    message = data.event.message
    if message.chat_type != "p2p":
        return None

    msg_type = str(message.message_type or "").strip().lower()
    if msg_type not in {"file", "image", "audio", "video", "media"}:
        return None

    payload = _parse_message_content(message.content or "{}")
    key_field = "image_key" if msg_type == "image" else "file_key"
    resource_key = str(payload.get(key_field, "")).strip()
    if not resource_key:
        raise RuntimeError(f"濞戝牊浼呯紓鍝勭毌 {key_field}")

    file_name_hint = str(
        payload.get("file_name")
        or payload.get("name")
        or payload.get("title")
        or ""
    ).strip()

    local_path, size_bytes, original_name = _download_message_resource(
        client,
        message_id=str(message.message_id or ""),
        resource_key=resource_key,
        resource_type=msg_type,
        file_name_hint=file_name_hint,
    )
    return (
        f"User uploaded a {msg_type} attachment via Feishu.\n"
        f"Original filename: {original_name}\n"
        f"Local path: {local_path}\n"
        f"Size: {size_bytes} bytes.\n"
        "Acknowledge receipt first, then process the file per user intent."
    )


async def _extract_user_input(data: P2ImMessageReceiveV1, client: lark.Client) -> Optional[str]:
    text = _extract_text_message(data)
    if text:
        return text
    return await asyncio.to_thread(_extract_file_like_message, data, client)



async def _handle_message(
    data: P2ImMessageReceiveV1,
    agent,
    client: lark.Client,
    tracker: MainLaneTracker,
    scheduler_store: SchedulerStore,
    agent_id: str,
    session_manager: SessionManager,
    memory_manager: runtime.MemoryManager | None,
    system_prompt: str = "",
) -> None:
    message = data.event.message
    sender_open_id = _safe_getattr(data, "event.sender.sender_id.open_id")
    sender_user_id = _safe_getattr(data, "event.sender.sender_id.user_id")
    print(
        "[feishu] inbound message"
        f" chat_id={message.chat_id}"
        f" message_id={message.message_id}"
        f" chat_type={message.chat_type}"
        f" msg_type={message.message_type}"
        f" sender_open_id={sender_open_id}"
        f" sender_user_id={sender_user_id}"
    )

    try:
        text = await _extract_user_input(data, client)
    except Exception as exc:
        print(f"[feishu] failed to parse inbound message: {exc}")
        try:
            _send_message(
                client,
                "chat_id",
                message.chat_id,
                "text",
                {"text": f"Failed to parse your message: {exc}"},
            )
        except Exception:
            pass
        return

    if not text:
        print(f"[feishu] ignored unsupported message type: {message.message_type}")
        return

    thread_id = message.chat_id
    user_id = sender_user_id or sender_open_id or ""
    owner_id = user_id or thread_id
    scheduler_store.upsert_last_user(
        agent_id=agent_id,
        channel=CHANNEL_FEISHU,
        thread_id=thread_id,
        chat_id=message.chat_id,
        chat_type=message.chat_type,
        user_id=user_id,
        updated_at=now_ms(),
    )
    config = {"configurable": {"thread_id": thread_id}}
    if runtime.log_tool_calls:
        config["callbacks"] = [runtime.ToolLogger()]

    token = _FEISHU_CONTEXT.set(
        _FeishuContext(
            client=client,
            chat_id=message.chat_id,
            chat_type=message.chat_type,
            sender_open_id=sender_open_id,
            sender_user_id=sender_user_id,
        )
    )
    try:
        session = session_manager.get_or_create(owner_id)
        session.add_message("user", text)
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt or FEISHU_MEDIA_TOOL_PROMPT},
                {"role": "user", "content": text},
            ]
        }
        print(f"[feishu] user input: {_truncate_text(text)}")
        async with tracker:
            result = await agent.ainvoke(payload, config=config)
        reply = result["messages"][-1].content
        reply_text = reply if isinstance(reply, str) else str(reply)
        print(f"[feishu] AI reply: {_truncate_text(reply_text)}")

        if str(reply_text).strip():
            ctx = _FEISHU_CONTEXT.get()
            if ctx is None:
                _send_message(client, "chat_id", message.chat_id, "text", {"text": reply_text})
            else:
                receive_type, receive_id = _select_receive_target(ctx)
                _send_message(ctx.client, receive_type, receive_id, "text", {"text": reply_text})

        session.add_message("assistant", reply_text)
        await asyncio.to_thread(session_manager.save, session)
        if memory_manager is not None:
            history = session.get_history()
            print(
                f"[feishu][memory] record turn owner_id={owner_id} "
                f"messages={len(history)} summary_every={runtime.memory_summary_every}"
            )
            await asyncio.to_thread(memory_manager.record_turn, owner_id, history)
        else:
            print(f"[feishu][memory] memory manager disabled owner_id={owner_id}")
    finally:
        _FEISHU_CONTEXT.reset(token)



def _start_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()



async def _build_agent():
    disable_mcp = os.getenv("FEISHU_DISABLE_MCP", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    original_mcp_servers = os.getenv("MCP_SERVERS")
    try:
        if disable_mcp:
            os.environ["MCP_SERVERS"] = "{}"
        agent = await runtime.build_agent(
            extra_tools=[
                feishu_send_media,
                cron_status,
                cron_list,
                cron_add,
                cron_remove,
                cron_run,
            ],
        )
    finally:
        if disable_mcp:
            if original_mcp_servers is None:
                os.environ.pop("MCP_SERVERS", None)
            else:
                os.environ["MCP_SERVERS"] = original_mcp_servers
    return agent


async def _summarize_startup_log_and_notify(
    agent,
    client: lark.Client,
    scheduler_store: SchedulerStore,
    *,
    agent_id: str,
    channel: str,
) -> None:
    log_tail = await asyncio.to_thread(_read_log_tail_words, LOG_FILE_PATH, LOG_SUMMARY_WORDS)
    if not log_tail:
        print("[log] startup summary skipped: log is empty")
        return

    target = scheduler_store.get_last_user(agent_id, channel)
    if target is None:
        print("[log] startup summary skipped: no last user target")
        return
    if target.chat_type != "p2p":
        print(f"[log] startup summary skipped: unsupported chat_type={target.chat_type}")
        return

    summary_prompt = (
        "你是同一个飞书助手。下面是你最近运行日志的最后部分。"
        "请先简要回顾你之前做过的事，再给出当前状态与下一步建议。"
        "要求：中文，120-220字，避免编造，避免输出代码块。"
    )
    payload = {
        "messages": [
            {"role": "system", "content": summary_prompt},
            {
                "role": "user",
                "content": (
                    f"以下是日志最后 {LOG_SUMMARY_WORDS} 词（可能不足）：\n\n"
                    f"{log_tail}"
                ),
            },
        ]
    }
    config = {"configurable": {"thread_id": target.thread_id or target.chat_id}}
    if runtime.log_tool_calls:
        config["callbacks"] = [runtime.ToolLogger()]

    try:
        result = await agent.ainvoke(payload, config=config)
        reply = result["messages"][-1].content
        summary_text = reply if isinstance(reply, str) else str(reply)
        summary_text = summary_text.strip()
        if not summary_text:
            print("[log] startup summary skipped: model returned empty text")
            return
        await asyncio.to_thread(
            _send_message,
            client,
            "chat_id",
            target.chat_id,
            "text",
            {"text": summary_text},
        )
        print(f"[log] startup summary sent to chat_id={target.chat_id}")
    except Exception as exc:
        print(f"[log] startup summary failed: {exc}")


class _FeishuConnectedHandler(logging.Handler):
    def __init__(self, on_connected):
        super().__init__(level=logging.INFO)
        self._on_connected = on_connected
        self._triggered = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._triggered:
            return
        try:
            message = record.getMessage()
        except Exception:
            return
        if "connected to wss://" not in str(message):
            return
        self._triggered = True
        try:
            self._on_connected()
        except Exception as exc:
            print(f"[log] failed to schedule startup summary: {exc}")


def _build_system_prompt() -> str:
    return FEISHU_MEDIA_TOOL_PROMPT

def main() -> None:
    app_id, app_secret = _setup_lark_env()
    agent = asyncio.run(_build_agent())

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_start_loop, args=(loop,), daemon=True)
    loop_thread.start()

    client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
    try:
        scheduler_store = SchedulerStore(SCHEDULER_DB_PATH)
    except Exception as exc:
        fallback_path = (
            Path(tempfile.gettempdir()) / "deepagent_runtime" / "scheduler_runtime.sqlite"
        )
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[scheduler] failed to open {SCHEDULER_DB_PATH}: {exc}; "
            f"fallback to {fallback_path}"
        )
        scheduler_store = SchedulerStore(fallback_path)
    tracker = MainLaneTracker()
    heartbeat_cfg = _build_heartbeat_config()
    workspace_root = _resolve_heartbeat_workspace_root()
    wake = HeartbeatWake()
    FEISHU_MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    session_manager = SessionManager(sessions_dir=FEISHU_MEMORY_ROOT / "sessions")
    memory_manager = _build_feishu_memory_manager()

    async def _invoke_agent_for_heartbeat(payload, config):
        invoke_config = dict(config or {})
        if runtime.log_tool_calls and "callbacks" not in invoke_config:
            invoke_config["callbacks"] = [runtime.ToolLogger()]
        result = await agent.ainvoke(payload, config=invoke_config)
        reply = result["messages"][-1].content
        return reply if isinstance(reply, str) else str(reply)

    async def _send_text_for_heartbeat(chat_id: str, text: str) -> None:
        await asyncio.to_thread(_send_message, client, "chat_id", chat_id, "text", {"text": text})

    heartbeat_deps = HeartbeatDeps(
        store=scheduler_store,
        agent_id=DEFAULT_AGENT_ID,
        channel=CHANNEL_FEISHU,
        workspace_root=workspace_root,
        is_busy=tracker.is_busy,
        invoke_agent=_invoke_agent_for_heartbeat,
        send_text=_send_text_for_heartbeat,
        enter_context=lambda chat_id, chat_type, user_id="": _enter_feishu_context(
            client,
            chat_id,
            chat_type,
            sender_user_id=user_id,
        ),
        exit_context=_exit_feishu_context,
    )

    runner = HeartbeatRunner(heartbeat_cfg, heartbeat_deps, wake)

    async def _run_heartbeat_now(reason: str):
        return await run_heartbeat_once(heartbeat_cfg, heartbeat_deps, reason=reason)

    cron_service = CronService(
        CronDeps(
            store=scheduler_store,
            agent_id=DEFAULT_AGENT_ID,
            channel=CHANNEL_FEISHU,
            request_heartbeat_now=lambda reason: wake.request_now(reason),
            run_heartbeat_once=_run_heartbeat_now,
        )
    )
    _bind_cron_service(cron_service)

    def _start_scheduler():
        runner.start()
        cron_service.start()

    loop.call_soon_threadsafe(_start_scheduler)
    if runtime.log_tool_calls:
        print("[feishu] tool call logging is enabled")

    system_prompt = _build_system_prompt()

    def _on_message(data: P2ImMessageReceiveV1) -> None:
        future = asyncio.run_coroutine_threadsafe(
            _handle_message(
                data,
                agent,
                client,
                tracker,
                scheduler_store,
                DEFAULT_AGENT_ID,
                session_manager,
                memory_manager,
                system_prompt,
            ),
            loop,
        )

        def _log_result(task: asyncio.Future) -> None:
            try:
                task.result()
            except Exception as exc:  # pragma: no cover - runtime logging only
                print(f"Handle message failed: {exc}")

        future.add_done_callback(_log_result)

    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(_on_message)
        .build()
    )

    ws_client = lark.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    def _schedule_startup_summary_once() -> None:
        future = asyncio.run_coroutine_threadsafe(
            _summarize_startup_log_and_notify(
                agent,
                client,
                scheduler_store,
                agent_id=DEFAULT_AGENT_ID,
                channel=CHANNEL_FEISHU,
            ),
            loop,
        )

        def _log_summary_result(task: asyncio.Future) -> None:
            try:
                task.result()
            except Exception as exc:
                print(f"[log] startup summary task failed: {exc}")

        future.add_done_callback(_log_summary_result)

    startup_summary_handler = _FeishuConnectedHandler(_schedule_startup_summary_once)
    root_logger = logging.getLogger()
    lark_logger = logging.getLogger("Lark")
    root_logger.addHandler(startup_summary_handler)
    lark_logger.addHandler(startup_summary_handler)

    try:
        print("Feishu websocket client started. Waiting for messages.")
        ws_client.start()
    except KeyboardInterrupt:
        print("Bot stopped by keyboard interrupt.")
    finally:
        root_logger.removeHandler(startup_summary_handler)
        lark_logger.removeHandler(startup_summary_handler)
        loop.call_soon_threadsafe(loop.stop)



if __name__ == "__main__":
    main()
