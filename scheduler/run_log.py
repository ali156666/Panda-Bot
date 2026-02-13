"""Cron job run log persistence system.

Stores execution history in JSONL format, one file per job.
Automatically prunes logs to prevent unbounded growth.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from .cron_types import CronEvent


class CronRunLog:
    """Manages cron job execution logs.
    
    Each job has its own log file: {log_dir}/runs/{job_id}.jsonl
    Logs are automatically pruned when they exceed size limits.
    """
    
    def __init__(
        self,
        log_dir: Path | str,
        max_bytes: int = 2_000_000,
        keep_lines: int = 2_000,
    ):
        """Initialize run log manager.
        
        Args:
            log_dir: Base directory for log files
            max_bytes: Maximum log file size before pruning (default: 2MB)
            keep_lines: Number of lines to keep after pruning (default: 2000)
        """
        self.log_dir = Path(log_dir)
        self.runs_dir = self.log_dir / "runs"
        self.max_bytes = max_bytes
        self.keep_lines = keep_lines
        self._lock = threading.Lock()
        self._pending_writes: dict[str, list[dict[str, Any]]] = {}
    
    def _ensure_dir(self) -> None:
        """Ensure the runs directory exists."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_path(self, job_id: str) -> Path:
        """Get the log file path for a job."""
        # Sanitize job_id to prevent path traversal
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)
        return self.runs_dir / f"{safe_id}.jsonl"
    
    def append(self, event: CronEvent) -> None:
        """Append a finished event to the job's log.
        
        Only 'finished' events are logged for history.
        
        Args:
            event: The cron event to log
        """
        if event.action != "finished":
            return
        
        entry = {
            "ts": event.run_at_ms or 0,
            "job_id": event.job_id,
            "action": event.action,
            "status": event.status,
            "error": event.error,
            "summary": event.summary,
            "run_at_ms": event.run_at_ms,
            "duration_ms": event.duration_ms,
            "next_run_at_ms": event.next_run_at_ms,
        }
        # Remove None values for cleaner logs
        entry = {k: v for k, v in entry.items() if v is not None}
        
        with self._lock:
            self._ensure_dir()
            log_path = self._log_path(event.job_id)
            
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                # Check if pruning is needed
                self._prune_if_needed(log_path)
            except OSError as e:
                print(f"[run_log] Failed to write log for {event.job_id}: {e}")
    
    def _prune_if_needed(self, log_path: Path) -> None:
        """Prune log file if it exceeds size limit."""
        try:
            stat = log_path.stat()
            if stat.st_size <= self.max_bytes:
                return
            
            # Read and keep only recent lines
            with log_path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            kept = lines[-self.keep_lines:] if len(lines) > self.keep_lines else lines
            
            # Atomic write using temp file
            tmp_path = log_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
            
            tmp_path.replace(log_path)
            
        except OSError:
            pass  # Ignore pruning errors
    
    def read(
        self,
        job_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Read recent log entries for a job.
        
        Args:
            job_id: The job ID to read logs for
            limit: Maximum number of entries to return (default: 200)
        
        Returns:
            List of log entries, newest last
        """
        limit = max(1, min(5000, limit))
        log_path = self._log_path(job_id)
        
        if not log_path.exists():
            return []
        
        try:
            with log_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        
        # Parse from end to respect limit
        entries: list[dict[str, Any]] = []
        for i in range(len(lines) - 1, -1, -1):
            if len(entries) >= limit:
                break
            
            line = lines[i].strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if obj.get("action") != "finished":
                    continue
                if not isinstance(obj.get("job_id"), str):
                    continue
                entries.append(obj)
            except json.JSONDecodeError:
                continue
        
        # Return in chronological order
        entries.reverse()
        return entries
    
    def read_all(self, limit: int = 200) -> list[dict[str, Any]]:
        """Read recent log entries from all jobs.
        
        Args:
            limit: Maximum number of entries per job (default: 200)
        
        Returns:
            List of all log entries, sorted by timestamp
        """
        if not self.runs_dir.exists():
            return []
        
        all_entries: list[dict[str, Any]] = []
        
        try:
            for log_file in self.runs_dir.glob("*.jsonl"):
                job_id = log_file.stem
                entries = self.read(job_id, limit=limit)
                all_entries.extend(entries)
        except OSError:
            pass
        
        # Sort by timestamp
        all_entries.sort(key=lambda e: e.get("ts", 0))
        return all_entries
    
    def clear(self, job_id: str) -> bool:
        """Clear all logs for a job.
        
        Args:
            job_id: The job ID to clear logs for
        
        Returns:
            True if logs were cleared, False otherwise
        """
        log_path = self._log_path(job_id)
        
        try:
            if log_path.exists():
                log_path.unlink()
                return True
        except OSError:
            pass
        
        return False
    
    def delete_job_logs(self, job_id: str) -> None:
        """Delete log file when a job is removed."""
        self.clear(job_id)
