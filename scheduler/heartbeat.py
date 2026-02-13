import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .store import SchedulerStore
from .utils import (
    HEARTBEAT_TOKEN,
    clamp_timeout_ms,
    is_heartbeat_content_effectively_empty,
    is_within_active_hours,
    now_ms,
    parse_duration_ms,
    strip_heartbeat_token,
)


DEFAULT_HEARTBEAT_PROMPT = (
    "请读取 HEARTBEAT.md（如果存在）并严格按其要求执行。"
    "不要臆测历史对话中的待办。"
    f"如果没有需要发送给用户的内容，请只回复 {HEARTBEAT_TOKEN}。"
)


@dataclass
class HeartbeatConfig:
    every: str = "30m"
    active_start: Optional[str] = None
    active_end: Optional[str] = None
    active_timezone: Optional[str] = None
    prompt: Optional[str] = None
    ack_max_chars: int = 300
    dedupe_window_ms: int = 24 * 60 * 60 * 1000


@dataclass
class HeartbeatRunResult:
    status: str
    reason: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class HeartbeatDeps:
    store: SchedulerStore
    agent_id: str
    channel: str
    workspace_root: Path
    is_busy: Callable[[], bool]
    invoke_agent: Callable[[dict[str, Any], dict[str, Any]], Awaitable[str]]
    send_text: Callable[[str, str], Awaitable[None]]
    enter_context: Optional[Callable[..., Any]] = None
    exit_context: Optional[Callable[[Any], None]] = None


def resolve_interval_ms(cfg: HeartbeatConfig) -> Optional[int]:
    return parse_duration_ms(cfg.every, default_unit="m")


def _build_prompt(
    cfg: HeartbeatConfig,
    heartbeat_text: Optional[str],
    system_events: list[dict[str, Any]],
) -> str:
    prompt = (cfg.prompt or "").strip() or DEFAULT_HEARTBEAT_PROMPT
    parts = [prompt, ""]
    if heartbeat_text:
        parts.append("HEARTBEAT.md 内容：")
        parts.append(heartbeat_text.strip())
        parts.append("")
    if system_events:
        parts.append("系统事件（按时间顺序）：")
        for evt in system_events:
            text = str(evt.get("text", "")).strip()
            if text:
                parts.append(f"- {text}")
        parts.append("")
    parts.append("请输出需要发送给用户的内容。")
    return "\n".join(parts).strip()


async def run_heartbeat_once(
    cfg: HeartbeatConfig,
    deps: HeartbeatDeps,
    reason: Optional[str] = None,
) -> HeartbeatRunResult:
    started_at = now_ms()
    interval_ms = resolve_interval_ms(cfg)
    if interval_ms is None:
        return HeartbeatRunResult(status="skipped", reason="disabled")

    if deps.is_busy():
        return HeartbeatRunResult(status="skipped", reason="requests-in-flight")

    if not is_within_active_hours(
        started_at, cfg.active_start, cfg.active_end, cfg.active_timezone
    ):
        return HeartbeatRunResult(status="skipped", reason="quiet-hours")

    target = deps.store.get_last_user(deps.agent_id, deps.channel)
    if target is None:
        return HeartbeatRunResult(status="skipped", reason="no-target")

    heartbeat_path = Path(deps.workspace_root) / deps.agent_id / "HEARTBEAT.md"
    heartbeat_text: Optional[str] = None
    heartbeat_empty = False
    try:
        heartbeat_text = heartbeat_path.read_text(encoding="utf-8")
        heartbeat_empty = is_heartbeat_content_effectively_empty(heartbeat_text)
    except FileNotFoundError:
        heartbeat_text = None
    except Exception:
        heartbeat_text = None

    system_events = deps.store.list_pending_system_events(
        deps.agent_id, deps.channel, target.thread_id
    )
    event_ids = [int(evt["id"]) for evt in system_events if evt.get("id") is not None]

    def _consume_events() -> None:
        if event_ids:
            deps.store.mark_system_events_consumed(event_ids)

    if heartbeat_empty and not system_events:
        return HeartbeatRunResult(status="skipped", reason="empty-heartbeat-file")
    prompt = _build_prompt(cfg, heartbeat_text, system_events)
    payload = {"messages": [{"role": "system", "content": prompt}]}
    config = {"configurable": {"thread_id": target.thread_id}}

    token = None
    if deps.enter_context:
        try:
            token = deps.enter_context(target.chat_id, target.chat_type, target.user_id)
        except TypeError:
            token = deps.enter_context(target.chat_id, target.chat_type)

    try:
        reply_text = await deps.invoke_agent(payload, config)
    except Exception as exc:
        return HeartbeatRunResult(status="failed", reason=str(exc))
    finally:
        if token is not None and deps.exit_context:
            deps.exit_context(token)

    stripped = strip_heartbeat_token(reply_text, max_ack_chars=cfg.ack_max_chars)
    if stripped.should_skip:
        _consume_events()
        return HeartbeatRunResult(status="ran", reason="ok-token")

    text = stripped.text.strip()
    if not text:
        _consume_events()
        return HeartbeatRunResult(status="ran", reason="empty")

    state = deps.store.get_heartbeat_state(deps.agent_id, deps.channel, target.thread_id)
    last_text = state.last_text.strip()
    last_sent_at = state.last_sent_at
    if (
        last_text
        and last_sent_at
        and text == last_text
        and started_at - last_sent_at < cfg.dedupe_window_ms
    ):
        _consume_events()
        return HeartbeatRunResult(status="ran", reason="duplicate")

    await deps.send_text(target.chat_id, text)
    _consume_events()
    deps.store.update_heartbeat_state(
        agent_id=deps.agent_id,
        channel=deps.channel,
        thread_id=target.thread_id,
        last_text=text,
        last_sent_at=started_at,
        updated_at=started_at,
    )

    return HeartbeatRunResult(status="ran", duration_ms=now_ms() - started_at)


class HeartbeatWake:
    def __init__(self) -> None:
        self._handler: Optional[Callable[[Optional[str]], Awaitable[HeartbeatRunResult]]] = None
        self._pending_reason: Optional[str] = None
        self._scheduled = False
        self._running = False
        self._timer_task: Optional[asyncio.Task] = None

    def set_handler(
        self, handler: Optional[Callable[[Optional[str]], Awaitable[HeartbeatRunResult]]]
    ) -> None:
        self._handler = handler
        if handler and self._pending_reason:
            self.request_now(self._pending_reason)

    def request_now(self, reason: Optional[str] = None, coalesce_ms: int = 250) -> None:
        self._pending_reason = reason or self._pending_reason or "requested"
        self._schedule(coalesce_ms)

    def _schedule(self, coalesce_ms: int) -> None:
        if self._timer_task:
            return
        self._timer_task = asyncio.create_task(self._run_later(coalesce_ms))

    async def _run_later(self, coalesce_ms: int) -> None:
        await asyncio.sleep(max(0, coalesce_ms) / 1000)
        self._timer_task = None
        self._scheduled = False
        handler = self._handler
        if handler is None:
            return
        if self._running:
            self._scheduled = True
            self._schedule(coalesce_ms)
            return
        reason = self._pending_reason
        self._pending_reason = None
        self._running = True
        try:
            result = await handler(reason)
            if result.status == "skipped" and result.reason == "requests-in-flight":
                self._pending_reason = reason or "retry"
                self._schedule(1000)
        except Exception:
            self._pending_reason = reason or "retry"
            self._schedule(1000)
        finally:
            self._running = False
            if self._pending_reason or self._scheduled:
                self._schedule(coalesce_ms)


class HeartbeatRunner:
    def __init__(self, cfg: HeartbeatConfig, deps: HeartbeatDeps, wake: HeartbeatWake):
        self._cfg = cfg
        self._deps = deps
        self._wake = wake
        self._stopped = False
        self._timer_task: Optional[asyncio.Task] = None
        self._next_due_ms: Optional[int] = None

    def start(self) -> None:
        self._wake.set_handler(self._run)
        self._schedule_next()

    def stop(self) -> None:
        self._stopped = True
        self._wake.set_handler(None)
        if self._timer_task:
            self._timer_task.cancel()
        self._timer_task = None

    async def _sleep_then_wake(self, delay_ms: int) -> None:
        await asyncio.sleep(delay_ms / 1000)
        self._wake.request_now(reason="interval", coalesce_ms=0)

    def _schedule_next(self) -> None:
        if self._stopped:
            return
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None
        interval_ms = resolve_interval_ms(self._cfg)
        if interval_ms is None:
            return
        now = now_ms()
        if self._next_due_ms is None or self._next_due_ms <= now:
            self._next_due_ms = now + interval_ms
        delay_ms = clamp_timeout_ms(self._next_due_ms - now)
        self._timer_task = asyncio.create_task(self._sleep_then_wake(delay_ms))

    async def _run(self, reason: Optional[str]) -> HeartbeatRunResult:
        if self._stopped:
            return HeartbeatRunResult(status="skipped", reason="disabled")
        interval_ms = resolve_interval_ms(self._cfg)
        if interval_ms is None:
            return HeartbeatRunResult(status="skipped", reason="disabled")
        now = now_ms()
        if reason == "interval" and self._next_due_ms and now < self._next_due_ms:
            return HeartbeatRunResult(status="skipped", reason="not-due")

        result = await run_heartbeat_once(self._cfg, self._deps, reason=reason)
        if result.status == "skipped" and result.reason == "requests-in-flight":
            return result

        self._next_due_ms = now + interval_ms
        self._schedule_next()
        return result
