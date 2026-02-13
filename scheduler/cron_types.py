"""Cron scheduling type definitions aligned with OpenClaw architecture."""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, TypedDict


# ============================================================================
# Schedule Types
# ============================================================================

class AtSchedule(TypedDict):
    """One-shot schedule at a specific timestamp."""
    kind: Literal["at"]
    at_ms: int


class EverySchedule(TypedDict, total=False):
    """Recurring schedule at fixed intervals."""
    kind: Literal["every"]
    every_ms: int
    anchor_ms: int  # Optional anchor point


class CronExprSchedule(TypedDict, total=False):
    """Cron expression schedule with optional timezone."""
    kind: Literal["cron"]
    expr: str
    tz: str  # Optional timezone


CronSchedule = AtSchedule | EverySchedule | CronExprSchedule


# ============================================================================
# Payload Types
# ============================================================================

class SystemEventPayload(TypedDict):
    """Payload for system event injection."""
    kind: Literal["systemEvent"]
    text: str


class AgentTurnPayload(TypedDict, total=False):
    """Payload for isolated agent turn execution."""
    kind: Literal["agentTurn"]
    message: str
    model: str  # Optional model override
    thinking: str  # Optional thinking level
    timeout_seconds: int  # Execution timeout
    deliver: bool  # Whether to deliver response
    channel: str  # Optional delivery channel
    to: str  # Optional recipient
    best_effort_deliver: bool


CronPayload = SystemEventPayload | AgentTurnPayload


# ============================================================================
# Isolation Config
# ============================================================================

class CronIsolation(TypedDict, total=False):
    """Configuration for isolated session execution."""
    post_to_main_prefix: str  # Prefix for summary posted to main
    post_to_main_mode: Literal["summary", "full"]  # What to post back
    post_to_main_max_chars: int  # Max chars for full mode (default: 8000)


# ============================================================================
# Job State
# ============================================================================

class CronJobState(TypedDict, total=False):
    """Runtime state of a cron job."""
    next_run_at: Optional[int]  # Next scheduled run timestamp (ms)
    running_at: Optional[int]  # Currently running since (ms)
    last_run_at: Optional[int]  # Last run timestamp (ms)
    last_status: Optional[Literal["ok", "error", "skipped"]]
    last_error: Optional[str]
    last_duration_ms: Optional[int]


# ============================================================================
# Cron Job
# ============================================================================

CronSessionTarget = Literal["main", "isolated"]
CronWakeMode = Literal["next-heartbeat", "now"]


class CronJob(TypedDict, total=False):
    """Full cron job definition."""
    job_id: str
    agent_id: str  # Optional agent ID
    name: str
    description: str
    enabled: bool
    delete_after_run: bool  # Delete one-shot jobs after completion
    created_at: int  # Creation timestamp (ms)
    updated_at: int  # Last update timestamp (ms)
    schedule: CronSchedule
    session_target: CronSessionTarget
    wake_mode: CronWakeMode
    payload: CronPayload
    isolation: CronIsolation
    state: CronJobState


class CronJobCreate(TypedDict, total=False):
    """Input for creating a new cron job."""
    name: str
    description: str
    enabled: bool
    delete_after_run: bool
    schedule: CronSchedule
    session_target: CronSessionTarget
    wake_mode: CronWakeMode
    payload: CronPayload
    isolation: CronIsolation
    agent_id: str


class CronJobPatch(TypedDict, total=False):
    """Input for updating an existing cron job."""
    name: str
    description: str
    enabled: bool
    delete_after_run: bool
    schedule: CronSchedule
    session_target: CronSessionTarget
    wake_mode: CronWakeMode
    payload: CronPayload
    isolation: CronIsolation


# ============================================================================
# Events
# ============================================================================

CronEventAction = Literal["added", "updated", "removed", "started", "finished"]


@dataclass
class CronEvent:
    """Event emitted during cron job lifecycle."""
    job_id: str
    action: CronEventAction
    run_at_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    status: Optional[Literal["ok", "error", "skipped"]] = None
    error: Optional[str] = None
    summary: Optional[str] = None
    next_run_at_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "action": self.action,
        }
        if self.run_at_ms is not None:
            result["run_at_ms"] = self.run_at_ms
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.status is not None:
            result["status"] = self.status
        if self.error is not None:
            result["error"] = self.error
        if self.summary is not None:
            result["summary"] = self.summary
        if self.next_run_at_ms is not None:
            result["next_run_at_ms"] = self.next_run_at_ms
        return result


# ============================================================================
# Service Results
# ============================================================================

@dataclass
class CronStatusSummary:
    """Summary of cron service status."""
    enabled: bool
    store_path: str
    jobs: int
    next_wake_at_ms: Optional[int]


@dataclass
class CronRunResult:
    """Result of manually running a job."""
    ok: bool
    ran: bool = False
    reason: Optional[str] = None


@dataclass
class CronRemoveResult:
    """Result of removing a job."""
    ok: bool
    removed: bool = False


# ============================================================================
# Callback Types
# ============================================================================

OnCronEventCallback = Callable[[CronEvent], None]


@dataclass
class IsolatedAgentResult:
    """Result from running an isolated agent turn."""
    status: Literal["ok", "error", "skipped"]
    summary: Optional[str] = None
    output_text: Optional[str] = None
    error: Optional[str] = None
