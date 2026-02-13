import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - 运行环境不支持时区库
    ZoneInfo = None


HEARTBEAT_TOKEN = "HEARTBEAT_OK"

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?\s*$")
_UNIT_MULTIPLIER = {
    "ms": 1,
    "s": 1000,
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
}


def parse_duration_ms(raw: str, default_unit: str = "m") -> Optional[int]:
    """解析 30m/1h 之类的间隔字符串。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    match = _DURATION_RE.match(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or default_unit).lower()
    if unit not in _UNIT_MULTIPLIER:
        return None
    ms = int(value * _UNIT_MULTIPLIER[unit])
    return ms if ms > 0 else None


def now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _get_local_tz():
    return datetime.now().astimezone().tzinfo or timezone.utc


def resolve_timezone(raw: Optional[str]) -> timezone:
    """解析时区字符串，失败时回退到本地时区。"""
    trimmed = (raw or "").strip()
    if not trimmed or trimmed.lower() == "local":
        return _get_local_tz()
    if ZoneInfo is None:
        return _get_local_tz()
    try:
        return ZoneInfo(trimmed)
    except Exception:
        return _get_local_tz()


_ACTIVE_TIME_RE = re.compile(r"^([01]\d|2[0-3]|24):([0-5]\d)$")


def _parse_active_minutes(value: Optional[str], allow_24: bool) -> Optional[int]:
    if not value:
        return None
    value = value.strip()
    if not _ACTIVE_TIME_RE.match(value):
        return None
    hour_str, minute_str = value.split(":")
    hour = int(hour_str)
    minute = int(minute_str)
    if hour == 24:
        if not allow_24 or minute != 0:
            return None
        return 24 * 60
    return hour * 60 + minute


def is_within_active_hours(
    now_epoch_ms: int,
    start: Optional[str],
    end: Optional[str],
    tz_name: Optional[str],
) -> bool:
    """判断当前时间是否在允许的活跃时段内。"""
    start_min = _parse_active_minutes(start, allow_24=False)
    end_min = _parse_active_minutes(end, allow_24=True)
    if start_min is None or end_min is None or start_min == end_min:
        return True
    tzinfo = resolve_timezone(tz_name)
    now_dt = datetime.fromtimestamp(now_epoch_ms / 1000, tz=tzinfo)
    current_min = now_dt.hour * 60 + now_dt.minute
    if end_min > start_min:
        return start_min <= current_min < end_min
    return current_min >= start_min or current_min < end_min


def is_heartbeat_content_effectively_empty(content: Optional[str]) -> bool:
    """判断 HEARTBEAT.md 是否只有注释或空行。"""
    if content is None:
        return False
    text = str(content)
    if not text.strip():
        return True
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if re.match(r"^#+(\s|$)", trimmed):
            continue
        if re.match(r"^[-*+]\s*(\[[\sXx]?\]\s*)?$", trimmed):
            continue
        return False
    return True


@dataclass
class StrippedHeartbeat:
    should_skip: bool
    text: str
    did_strip: bool


def _strip_token_edges(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if HEARTBEAT_TOKEN not in text:
        return text, False
    did_strip = False
    changed = True
    while changed:
        changed = False
        trimmed = text.strip()
        if trimmed.startswith(HEARTBEAT_TOKEN):
            text = trimmed[len(HEARTBEAT_TOKEN) :].lstrip()
            did_strip = True
            changed = True
        if trimmed.endswith(HEARTBEAT_TOKEN):
            text = trimmed[: -len(HEARTBEAT_TOKEN)].rstrip()
            did_strip = True
            changed = True
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed, did_strip


def strip_heartbeat_token(raw: Optional[str], max_ack_chars: int = 300) -> StrippedHeartbeat:
    """剥离 HEARTBEAT_OK 标记，决定是否应跳过发送。"""
    if raw is None:
        return StrippedHeartbeat(True, "", False)
    text = str(raw).strip()
    if not text:
        return StrippedHeartbeat(True, "", False)
    normalized = (
        re.sub(r"<[^>]*>", " ", text)
        .replace("&nbsp;", " ")
        .strip("*`~_ ")
        .strip()
    )
    if HEARTBEAT_TOKEN not in text and HEARTBEAT_TOKEN not in normalized:
        return StrippedHeartbeat(False, text, False)
    stripped_text, did_strip = _strip_token_edges(text)
    if not did_strip:
        stripped_text, did_strip = _strip_token_edges(normalized)
    if not did_strip:
        return StrippedHeartbeat(False, text, False)
    if not stripped_text:
        return StrippedHeartbeat(True, "", True)
    if len(stripped_text) <= max(0, int(max_ack_chars)):
        return StrippedHeartbeat(True, "", True)
    return StrippedHeartbeat(False, stripped_text, True)


def clamp_timeout_ms(delay_ms: int, max_ms: int = 2**31 - 1) -> int:
    """限制超长定时器的延迟，避免溢出。"""
    return max(0, min(int(delay_ms), int(max_ms)))
