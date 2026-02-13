"""Enhanced Cron scheduling service with event broadcasting, CRUD API, and agentTurn support.

Based on OpenClaw architecture with Python/asyncio implementation.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from croniter import croniter

from .cron_types import (
    CronEvent,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronRemoveResult,
    CronRunResult,
    CronStatusSummary,
    IsolatedAgentResult,
    OnCronEventCallback,
)
from .run_log import CronRunLog
from .store import SchedulerStore
from .utils import clamp_timeout_ms, now_ms, resolve_timezone


# 调度执行保护参数（避免任务卡死）
HEARTBEAT_NOW_CALL_TIMEOUT_SEC = 8.0
HEARTBEAT_NOW_MAX_WAIT_MS = 20_000
HEARTBEAT_NOW_RETRY_SLEEP_SEC = 0.25
JOB_EXEC_TIMEOUT_SEC = 90.0
STALE_RUNNING_RESET_MS = 10 * 60 * 1000


def compute_next_run_at_ms(schedule: dict[str, Any], base_ms: int) -> Optional[int]:
    """Compute next run timestamp based on schedule type."""
    kind = str(schedule.get("kind", "")).strip()
    if kind == "at":
        at_ms = int(schedule.get("at_ms", 0))
        return at_ms if at_ms > base_ms else None
    if kind == "every":
        every_ms = int(schedule.get("every_ms", 0))
        anchor_ms = int(schedule.get("anchor_ms", base_ms))
        if every_ms <= 0:
            return None
        if base_ms < anchor_ms:
            return anchor_ms
        elapsed = base_ms - anchor_ms
        steps = max(1, int((elapsed + every_ms - 1) / every_ms))
        return anchor_ms + steps * every_ms
    if kind == "cron":
        expr = str(schedule.get("expr", "")).strip()
        if not expr:
            return None
        tz_name = schedule.get("tz")
        tzinfo = resolve_timezone(tz_name)
        try:
            base_dt = datetime.fromtimestamp(base_ms / 1000, tz=tzinfo)
            next_dt = croniter(expr, base_dt).get_next(datetime)
        except Exception:
            return None
        if not next_dt:
            return None
        return int(next_dt.timestamp() * 1000)
    return None


def compute_initial_next_run_at_ms(job: dict[str, Any], now: int) -> Optional[int]:
    """Compute initial next_run_at for legacy rows with empty state.

    For manually inserted rows (bypassing CronService.add), this uses updated_at as
    the base timestamp when available so a just-missed run can still be caught up
    on the next scheduler tick.
    """
    base_ms = now
    updated_at = job.get("updated_at")
    if isinstance(updated_at, int) and 0 < updated_at < now:
        base_ms = updated_at
    return compute_next_run_at_ms(job.get("schedule", {}), base_ms)


@dataclass
class CronDeps:
    """Dependencies for CronService."""
    store: SchedulerStore
    agent_id: str
    channel: str
    request_heartbeat_now: Callable[[str], None]
    run_heartbeat_once: Optional[Callable[[str], Any]] = None
    run_isolated_agent: Optional[Callable[[dict, str], Any]] = None
    on_event: Optional[OnCronEventCallback] = None
    log_dir: Optional[Path] = None
    max_concurrent_runs: int = 5
    cron_enabled: bool = True


class CronService:
    """Enhanced cron scheduling service.
    
    Features:
    - Event broadcasting for job lifecycle
    - CRUD API for job management
    - agentTurn payload support for isolated execution
    - Concurrency control
    - Run log persistence
    """
    
    def __init__(self, deps: CronDeps) -> None:
        self._deps = deps
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._lock = asyncio.Lock()
        self._running_jobs: set[str] = set()
        # Used to detect stale running markers left by a previous process after restart.
        self._started_at_ms = now_ms()
        self._run_log: Optional[CronRunLog] = None
        
        if deps.log_dir:
            self._run_log = CronRunLog(deps.log_dir)
    
    # ========================================================================
    # Lifecycle
    # ========================================================================
    
    def start(self) -> None:
        """Start the cron service loop."""
        if self._task:
            return
        self._stopped = False
        self._task = asyncio.create_task(self._run_loop())
    
    def stop(self) -> None:
        """Stop the cron service."""
        self._stopped = True
        if self._task:
            self._task.cancel()
        self._task = None
    
    async def status(self) -> CronStatusSummary:
        """Get service status summary."""
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
            enabled_jobs = [j for j in jobs if j.get("enabled", True)]
            
            next_wake: Optional[int] = None
            for job in enabled_jobs:
                state = job.get("state", {}) or {}
                next_run = state.get("next_run_at")
                if isinstance(next_run, int):
                    if next_wake is None or next_run < next_wake:
                        next_wake = next_run
            
            return CronStatusSummary(
                enabled=self._deps.cron_enabled,
                store_path=str(self._deps.store.db_path),
                jobs=len(jobs),
                next_wake_at_ms=next_wake,
            )
    
    # ========================================================================
    # CRUD API
    # ========================================================================
    
    async def list(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        """List all cron jobs."""
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
            if not include_disabled:
                jobs = [j for j in jobs if j.get("enabled", True)]
            return jobs
    
    async def add(self, input_job: CronJobCreate) -> dict[str, Any]:
        """Add a new cron job."""
        now = now_ms()
        job_id = str(uuid.uuid4())
        
        job: dict[str, Any] = {
            "job_id": job_id,
            "name": input_job.get("name", "Unnamed Job"),
            "description": input_job.get("description", ""),
            "enabled": input_job.get("enabled", True),
            "delete_after_run": input_job.get("delete_after_run", False),
            "created_at": now,
            "updated_at": now,
            "schedule": input_job.get("schedule", {}),
            "schedule_kind": input_job.get("schedule", {}).get("kind", ""),
            "session_target": input_job.get("session_target", "main"),
            "wake_mode": input_job.get("wake_mode", "next-heartbeat"),
            "payload_kind": input_job.get("payload", {}).get("kind", "systemEvent"),
            "payload": input_job.get("payload", {}),
            "isolation": input_job.get("isolation", {}),
            "agent_id": input_job.get("agent_id", self._deps.agent_id),
            "state": {},
        }
        
        # Compute initial next_run_at
        next_run = compute_next_run_at_ms(job["schedule"], now)
        job["state"]["next_run_at"] = next_run
        
        async with self._lock:
            self._deps.store.upsert_cron_job(job)
        
        self._emit(CronEvent(job_id=job_id, action="added"))
        return job
    
    async def update(self, job_id: str, patch: CronJobPatch) -> Optional[dict[str, Any]]:
        """Update an existing cron job."""
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
            job = next((j for j in jobs if j.get("job_id") == job_id), None)
            
            if job is None:
                return None
            
            now = now_ms()
            
            # Apply patch
            if "name" in patch:
                job["name"] = patch["name"]
            if "description" in patch:
                job["description"] = patch["description"]
            if "enabled" in patch:
                job["enabled"] = patch["enabled"]
            if "delete_after_run" in patch:
                job["delete_after_run"] = patch["delete_after_run"]
            if "schedule" in patch:
                job["schedule"] = patch["schedule"]
                job["schedule_kind"] = patch["schedule"].get("kind", "")
            if "session_target" in patch:
                job["session_target"] = patch["session_target"]
            if "wake_mode" in patch:
                job["wake_mode"] = patch["wake_mode"]
            if "payload" in patch:
                job["payload"] = patch["payload"]
                job["payload_kind"] = patch["payload"].get("kind", "systemEvent")
            if "isolation" in patch:
                job["isolation"] = patch["isolation"]
            
            job["updated_at"] = now
            
            # Recompute next_run_at if schedule changed
            if "schedule" in patch:
                state = job.get("state", {}) or {}
                state["next_run_at"] = compute_next_run_at_ms(job["schedule"], now)
                job["state"] = state
            
            self._deps.store.upsert_cron_job(job)
        
        self._emit(CronEvent(job_id=job_id, action="updated"))
        return job
    
    async def remove(self, job_id: str) -> CronRemoveResult:
        """Remove a cron job."""
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
            job = next((j for j in jobs if j.get("job_id") == job_id), None)
            
            if job is None:
                return CronRemoveResult(ok=False, removed=False)
            
            # Remove from store (mark as disabled and delete)
            job["enabled"] = False
            self._deps.store.upsert_cron_job(job)
            
            # Clean up run logs
            if self._run_log:
                self._run_log.delete_job_logs(job_id)
        
        self._emit(CronEvent(job_id=job_id, action="removed"))
        return CronRemoveResult(ok=True, removed=True)
    
    async def run(self, job_id: str, mode: str = "due") -> CronRunResult:
        """Manually run a job.
        
        Args:
            job_id: Job to run
            mode: 'due' (only if due) or 'force' (run regardless)
        """
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
            job = next((j for j in jobs if j.get("job_id") == job_id), None)
            
            if job is None:
                return CronRunResult(ok=False, reason="job not found")
            
            now = now_ms()
            state = job.get("state", {}) or {}
            next_run = state.get("next_run_at")
            
            if mode == "due" and isinstance(next_run, int) and now < next_run:
                return CronRunResult(ok=True, ran=False, reason="not-due")
        
        await self._execute_job(job, now_ms(), forced=(mode == "force"))
        return CronRunResult(ok=True, ran=True)
    
    # ========================================================================
    # Event System
    # ========================================================================
    
    def _emit(self, event: CronEvent) -> None:
        """Emit a cron event."""
        try:
            if self._deps.on_event:
                self._deps.on_event(event)
            
            # Log finished events
            if event.action == "finished" and self._run_log:
                self._run_log.append(event)
        except Exception:
            pass  # Ignore event handler errors
    
    # ========================================================================
    # Scheduler Loop
    # ========================================================================
    
    async def _run_loop(self) -> None:
        """Main scheduling loop."""
        while not self._stopped:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[cron] tick error: {e}")
                await asyncio.sleep(1)
    
    async def _tick(self) -> None:
        """Single tick of the scheduler."""
        if not self._deps.cron_enabled:
            await asyncio.sleep(1)
            return
        
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
            now = now_ms()
            next_due: Optional[int] = None
            
            for job in jobs:
                if not job.get("enabled", True):
                    continue
                state = job.get("state", {}) or {}
                next_run = state.get("next_run_at")
                if not isinstance(next_run, int):
                    next_run = compute_initial_next_run_at_ms(job, now)
                    state["next_run_at"] = next_run
                    self._deps.store.update_cron_state(job["job_id"], state, now)
                if isinstance(next_run, int):
                    if next_due is None or next_run < next_due:
                        next_due = next_run
            
            if next_due is None:
                await asyncio.sleep(1)
                return
            
            # 注意：新的任务可能在 sleep 期间被写入数据库（例如外部直接写库）。
            # 为避免睡太久导致错过更早的新任务，这里限制单次 sleep 上限。
            delay_ms = min(clamp_timeout_ms(next_due - now), 5000)
        
        await asyncio.sleep(delay_ms / 1000)
        await self._run_due_jobs()
    
    async def _run_due_jobs(self) -> None:
        """Run all due jobs respecting concurrency limit."""
        async with self._lock:
            jobs = self._deps.store.list_cron_jobs()
        
        now = now_ms()
        for job in jobs:
            # Check concurrency limit
            if len(self._running_jobs) >= self._deps.max_concurrent_runs:
                break
            
            if not job.get("enabled", True):
                continue
            
            job_id = job.get("job_id")
            state = job.get("state", {}) or {}

            if job_id in self._running_jobs:
                continue

            # 兜底：跨重启或异常导致 running_at 长时间不清理时，自动解锁。
            running_at = state.get("running_at")
            if isinstance(running_at, int) and running_at > 0:
                # If running_at is older than current process start time, it came from a
                # previous process and should be recovered immediately.
                if running_at < self._started_at_ms:
                    state["running_at"] = None
                    state["last_error"] = "stale-running-recovered-after-restart"
                    self._deps.store.update_cron_state(job_id, state, now)
                    print(f"[cron] recovered stale running marker after restart: job_id={job_id}")
                elif now - running_at > STALE_RUNNING_RESET_MS:
                    state["running_at"] = None
                    state["last_error"] = "stale-running-reset"
                    self._deps.store.update_cron_state(job_id, state, now)
                    print(f"[cron] recovered stale running marker by timeout: job_id={job_id}")
                else:
                    continue

            next_run = state.get("next_run_at")
            if not isinstance(next_run, int) or now < next_run:
                continue

            await self._execute_job(job, now, forced=False)
    
    async def _execute_job(self, job: dict[str, Any], now: int, forced: bool = False) -> None:
        """Execute a single job."""
        job_id = job["job_id"]
        state = job.get("state", {}) or {}
        
        # Mark as running
        self._running_jobs.add(job_id)
        state["running_at"] = now
        state["last_error"] = None
        self._deps.store.update_cron_state(job_id, state, now)
        
        # Emit started event
        self._emit(CronEvent(job_id=job_id, action="started", run_at_ms=now))
        
        status = "ok"
        error: Optional[str] = None
        summary: Optional[str] = None
        deleted = False
        
        async def _do_execute() -> None:
            nonlocal status, error, summary

            session_target = job.get("session_target", "main")
            payload_kind = job.get("payload_kind", "systemEvent")
            payload = job.get("payload", {}) or {}

            if session_target == "main":
                # Main session: systemEvent only
                if payload_kind != "systemEvent":
                    status = "skipped"
                    error = "main job requires payload.kind=systemEvent"
                else:
                    text = str(payload.get("text", "")).strip()
                    if not text:
                        status = "skipped"
                        error = "systemEvent text is empty"
                    else:
                        target = self._deps.store.get_last_user(
                            self._deps.agent_id, self._deps.channel
                        )
                        if target is None:
                            status = "skipped"
                            error = "no-target"
                        else:
                            self._deps.store.add_system_event(
                                agent_id=self._deps.agent_id,
                                channel=self._deps.channel,
                                thread_id=target.thread_id,
                                text=text,
                                created_at=now,
                            )
                            summary = text
                            reason = f"cron:{job_id}"
                            if job.get("wake_mode") == "now" and self._deps.run_heartbeat_once:
                                await self._run_now(reason)
                            else:
                                self._deps.request_heartbeat_now(reason)

            elif session_target == "isolated":
                # Isolated session: agentTurn supported
                if payload_kind == "agentTurn":
                    if self._deps.run_isolated_agent:
                        message = str(payload.get("message", "")).strip()
                        if not message:
                            status = "skipped"
                            error = "agentTurn message is empty"
                        else:
                            result = await self._deps.run_isolated_agent(job, message)
                            if isinstance(result, IsolatedAgentResult):
                                status = result.status
                                error = result.error
                                summary = result.summary
                            else:
                                status = "ok"
                                summary = str(result)

                            # Post to main session if configured
                            isolation = job.get("isolation", {}) or {}
                            if status == "ok":
                                self._post_to_main(job_id, isolation, summary)
                    else:
                        status = "skipped"
                        error = "run_isolated_agent not configured"
                elif payload_kind == "systemEvent":
                    # Isolated session with systemEvent - just enqueue
                    text = str(payload.get("text", "")).strip()
                    if text:
                        self._deps.request_heartbeat_now(f"cron:{job_id}")
                        summary = text
                    else:
                        status = "skipped"
                        error = "systemEvent text is empty"
                else:
                    status = "skipped"
                    error = f"unsupported payload kind: {payload_kind}"
            else:
                status = "skipped"
                error = f"unsupported session target: {session_target}"

        try:
            await asyncio.wait_for(_do_execute(), timeout=JOB_EXEC_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            status = "error"
            error = f"job-exec-timeout>{JOB_EXEC_TIMEOUT_SEC}s"
        except Exception as exc:
            status = "error"
            error = str(exc)
        
        finally:
            self._running_jobs.discard(job_id)
        
        finished_at = now_ms()
        duration_ms = max(0, finished_at - now)
        
        # Handle one-shot jobs
        schedule_kind = job.get("schedule", {}).get("kind")
        should_delete = (
            schedule_kind == "at" and 
            status == "ok" and 
            job.get("delete_after_run", False)
        )
        
        if should_delete:
            async with self._lock:
                # Mark for deletion
                job["enabled"] = False
                self._deps.store.upsert_cron_job(job)
                deleted = True
                if self._run_log:
                    self._run_log.delete_job_logs(job_id)
        elif schedule_kind == "at" and status == "ok":
            # One-shot completed: disable but don't delete
            job["enabled"] = False
            state["next_run_at"] = None
        else:
            # Compute next run
            state["next_run_at"] = compute_next_run_at_ms(job.get("schedule", {}), finished_at)
        
        # Update state
        state["running_at"] = None
        state["last_run_at"] = now
        state["last_status"] = status
        state["last_error"] = error
        state["last_duration_ms"] = duration_ms
        
        self._deps.store.update_cron_state(job_id, state, finished_at)
        
        # Emit finished event
        self._emit(CronEvent(
            job_id=job_id,
            action="finished",
            run_at_ms=now,
            duration_ms=duration_ms,
            status=status,
            error=error,
            summary=summary,
            next_run_at_ms=state.get("next_run_at"),
        ))
        
        if deleted:
            self._emit(CronEvent(job_id=job_id, action="removed"))
    
    def _post_to_main(self, job_id: str, isolation: dict, summary: Optional[str]) -> None:
        """Post result summary to main session."""
        if not summary:
            return
        
        prefix = str(isolation.get("post_to_main_prefix", "Cron")).strip() or "Cron"
        mode = isolation.get("post_to_main_mode", "summary")
        max_chars = int(isolation.get("post_to_main_max_chars", 8000))
        
        body = summary
        if mode == "full" and len(body) > max_chars:
            body = body[:max_chars] + "…"
        
        text = f"{prefix}: {body}"
        
        try:
            target = self._deps.store.get_last_user(self._deps.agent_id, self._deps.channel)
            if target:
                self._deps.store.add_system_event(
                    agent_id=self._deps.agent_id,
                    channel=self._deps.channel,
                    thread_id=target.thread_id,
                    text=text,
                    created_at=now_ms(),
                )
                self._deps.request_heartbeat_now(f"cron:{job_id}:post")
        except Exception:
            pass
    
    async def _run_now(self, reason: str) -> None:
        """Run heartbeat immediately with bounded waits.

        防止 run_heartbeat_once 在 busy 场景下无限循环导致 cron job 卡死。
        """
        if not self._deps.run_heartbeat_once:
            self._deps.request_heartbeat_now(reason)
            return

        start = now_ms()

        while True:
            try:
                result = await asyncio.wait_for(
                    self._deps.run_heartbeat_once(reason),
                    timeout=HEARTBEAT_NOW_CALL_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                # 单次调用超时：降级异步触发，避免阻塞 job 收尾
                self._deps.request_heartbeat_now(reason)
                return
            except Exception:
                # 运行异常：降级异步触发
                self._deps.request_heartbeat_now(reason)
                return

            status = getattr(result, "status", None)
            skip_reason = getattr(result, "reason", "")

            # 非 requests-in-flight，视为本次处理完成
            if status != "skipped" or skip_reason != "requests-in-flight":
                return

            # busy 持续过久则降级异步触发，避免卡死
            if now_ms() - start > HEARTBEAT_NOW_MAX_WAIT_MS:
                self._deps.request_heartbeat_now(reason)
                return

            await asyncio.sleep(HEARTBEAT_NOW_RETRY_SLEEP_SEC)
    
    # ========================================================================
    # Run Log Access
    # ========================================================================
    
    def get_run_logs(self, job_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Get execution history for a job."""
        if not self._run_log:
            return []
        return self._run_log.read(job_id, limit=limit)
    
    def get_all_run_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        """Get execution history for all jobs."""
        if not self._run_log:
            return []
        return self._run_log.read_all(limit=limit)
