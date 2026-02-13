import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class LastUserTarget:
    agent_id: str
    channel: str
    thread_id: str
    chat_id: str
    chat_type: str
    user_id: str
    updated_at: int


@dataclass
class HeartbeatState:
    last_text: str
    last_sent_at: Optional[int]


class SchedulerStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = self._open()
        self._ensure_schema()

    def _open(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS last_user (
                    agent_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    chat_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (agent_id, channel, thread_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS heartbeat_state (
                    agent_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    last_text TEXT,
                    last_sent_at INTEGER,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (agent_id, channel, thread_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    job_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,
                    schedule_kind TEXT NOT NULL,
                    schedule_json TEXT NOT NULL,
                    payload_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    session_target TEXT NOT NULL,
                    wake_mode TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )

    def upsert_last_user(
        self,
        *,
        agent_id: str,
        channel: str,
        thread_id: str,
        chat_id: str,
        chat_type: str,
        user_id: str,
        updated_at: int,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO last_user
                    (agent_id, channel, thread_id, chat_id, chat_type, user_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, channel, thread_id)
                DO UPDATE SET
                    chat_id=excluded.chat_id,
                    chat_type=excluded.chat_type,
                    user_id=excluded.user_id,
                    updated_at=excluded.updated_at
                """,
                (agent_id, channel, thread_id, chat_id, chat_type, user_id, updated_at),
            )

    def get_last_user(self, agent_id: str, channel: str) -> Optional[LastUserTarget]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT agent_id, channel, thread_id, chat_id, chat_type, user_id, updated_at
                FROM last_user
                WHERE agent_id = ? AND channel = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (agent_id, channel),
            ).fetchone()
        if not row:
            return None
        return LastUserTarget(
            agent_id=row["agent_id"],
            channel=row["channel"],
            thread_id=row["thread_id"],
            chat_id=row["chat_id"],
            chat_type=row["chat_type"],
            user_id=row["user_id"],
            updated_at=int(row["updated_at"]),
        )

    def get_heartbeat_state(
        self, agent_id: str, channel: str, thread_id: str
    ) -> HeartbeatState:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT last_text, last_sent_at
                FROM heartbeat_state
                WHERE agent_id = ? AND channel = ? AND thread_id = ?
                """,
                (agent_id, channel, thread_id),
            ).fetchone()
        if not row:
            return HeartbeatState(last_text="", last_sent_at=None)
        last_text = row["last_text"] or ""
        last_sent_at = row["last_sent_at"]
        return HeartbeatState(last_text=str(last_text), last_sent_at=last_sent_at)

    def update_heartbeat_state(
        self,
        *,
        agent_id: str,
        channel: str,
        thread_id: str,
        last_text: str,
        last_sent_at: int,
        updated_at: int,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO heartbeat_state
                    (agent_id, channel, thread_id, last_text, last_sent_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, channel, thread_id)
                DO UPDATE SET
                    last_text=excluded.last_text,
                    last_sent_at=excluded.last_sent_at,
                    updated_at=excluded.updated_at
                """,
                (agent_id, channel, thread_id, last_text, last_sent_at, updated_at),
            )

    def add_system_event(
        self,
        *,
        agent_id: str,
        channel: str,
        thread_id: str,
        text: str,
        created_at: int,
    ) -> int:
        with self._lock, self._conn:
            existing = self._conn.execute(
                """
                SELECT id
                FROM system_events
                WHERE agent_id = ?
                  AND channel = ?
                  AND thread_id = ?
                  AND text = ?
                  AND status = 'pending'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_id, channel, thread_id, text),
            ).fetchone()
            if existing:
                return int(existing["id"])

            cur = self._conn.execute(
                """
                INSERT INTO system_events
                    (agent_id, channel, thread_id, text, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)
                """,
                (agent_id, channel, thread_id, text, created_at),
            )
            return int(cur.lastrowid)

    def list_pending_system_events(
        self, agent_id: str, channel: str, thread_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, text, created_at
                FROM system_events
                WHERE agent_id = ? AND channel = ? AND thread_id = ? AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (agent_id, channel, thread_id, int(limit)),
            ).fetchall()
        return [
            {"id": int(row["id"]), "text": row["text"], "created_at": int(row["created_at"])}
            for row in rows
        ]

    def mark_system_events_consumed(self, ids: Iterable[int]) -> None:
        id_list = [int(i) for i in ids if i is not None]
        if not id_list:
            return
        placeholders = ",".join(["?"] * len(id_list))
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE system_events SET status='consumed' WHERE id IN ({placeholders})",
                id_list,
            )

    def list_cron_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT job_id, enabled, schedule_kind, schedule_json, payload_kind,
                       payload_json, session_target, wake_mode, state_json, updated_at
                FROM cron_jobs
                """
            ).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            jobs.append(
                {
                    "job_id": row["job_id"],
                    "enabled": bool(row["enabled"]),
                    "schedule_kind": row["schedule_kind"],
                    "schedule": json.loads(row["schedule_json"] or "{}"),
                    "payload_kind": row["payload_kind"],
                    "payload": json.loads(row["payload_json"] or "{}"),
                    "session_target": row["session_target"],
                    "wake_mode": row["wake_mode"],
                    "state": json.loads(row["state_json"] or "{}"),
                    "updated_at": int(row["updated_at"]),
                }
            )
        return jobs

    def upsert_cron_job(self, job: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO cron_jobs
                    (job_id, enabled, schedule_kind, schedule_json, payload_kind,
                     payload_json, session_target, wake_mode, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id)
                DO UPDATE SET
                    enabled=excluded.enabled,
                    schedule_kind=excluded.schedule_kind,
                    schedule_json=excluded.schedule_json,
                    payload_kind=excluded.payload_kind,
                    payload_json=excluded.payload_json,
                    session_target=excluded.session_target,
                    wake_mode=excluded.wake_mode,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    job.get("job_id"),
                    1 if job.get("enabled", True) else 0,
                    job.get("schedule_kind"),
                    json.dumps(job.get("schedule", {}), ensure_ascii=False),
                    job.get("payload_kind"),
                    json.dumps(job.get("payload", {}), ensure_ascii=False),
                    job.get("session_target", "main"),
                    job.get("wake_mode", "next-heartbeat"),
                    json.dumps(job.get("state", {}), ensure_ascii=False),
                    int(job.get("updated_at", 0)),
                ),
            )

    def update_cron_state(self, job_id: str, state: dict[str, Any], updated_at: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE cron_jobs SET state_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(state, ensure_ascii=False), int(updated_at), job_id),
            )
