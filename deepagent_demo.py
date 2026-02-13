import asyncio
import json
import os
import shutil
import subprocess
import locale
import sys
import uuid
import inspect
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Literal
from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import EditResult, FilesystemBackend, perform_string_replacement
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool as lc_tool
from pydantic import BaseModel, Field
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    TodoListMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

try:
    import sqlite_vec
except Exception as exc:  # pragma: no cover - optional dependency
    sqlite_vec = None
    _sqlite_vec_import_error = exc

from tool import get_gemini_image_tools, get_tts_tools
try:
    from browser import get_browser_tools
except Exception:  # pragma: no cover - optional dependency
    get_browser_tools = None


try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:  # pragma: no cover - optional dependency
    MultiServerMCPClient = None


def load_env_file(path: str = ".env") -> None:
    """Minimal .env loader; always overwrites existing env vars."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ[key] = value


def require_env(names: list[str]) -> dict[str, str]:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print("Missing required environment variables:", ", ".join(missing))
        print("Tip: create a .env file next to this script or set them in your shell.")
        sys.exit(1)
    return {n: os.environ[n] for n in names}


load_env_file()

env = require_env(["OPENAI_API_KEY", "TAVILY_API_KEY"])
openai_api_key = env["OPENAI_API_KEY"]
tavily_api_key = env["TAVILY_API_KEY"]

openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

tavily_client = TavilyClient(api_key=tavily_api_key)
log_tool_calls = os.getenv("LOG_TOOL_CALLS", "1") != "0"
sandbox_timeout = int(os.getenv("SANDBOX_TIMEOUT", "20"))
# 已移除沙盒机制：始终允许本地执行与全路径访问。
allow_local_shell = True
# 启用本地终端执行时的根目录（仅影响相对路径与默认工作目录）。
local_shell_root = os.getenv("LOCAL_SHELL_ROOT", ".")
exec_output_encoding = os.getenv("EXEC_OUTPUT_ENCODING")
if not exec_output_encoding:
    try:
        exec_output_encoding = locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        exec_output_encoding = "utf-8"
skills_dir = os.getenv("SKILLS_DIR", "skills")
history_root = os.getenv("HISTORY_ROOT", ".deepagents_fs")
os.makedirs(history_root, exist_ok=True)
prompts_dir = Path(os.getenv("PROMPTS_DIR", "prompts")).resolve()
memory_root = Path(os.getenv("MEMORY_ROOT", "memory")).resolve()
memory_root.mkdir(parents=True, exist_ok=True)
memory_db_path = Path(
    os.getenv("MEMORY_DB_PATH", str(memory_root / "memory.sqlite"))
).resolve()
memory_max_messages = int(os.getenv("MEMORY_MAX_MESSAGES", "20"))
memory_summary_model_name = os.getenv("MEMORY_SUMMARY_MODEL", openai_model)
memory_summary_api_key = (
    os.getenv("MEMORY_SUMMARY_KEY")
    or os.getenv("MEMORY_SUMMARY_API_KEY")
    or openai_api_key
)
memory_embedding_model_name = os.getenv("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
memory_embedding_api_key = (
    os.getenv("MEMORY_EMBEDDING_KEY")
    or os.getenv("MEMORY_EMBEDDING_API_KEY")
    or openai_api_key
)
memory_embedding_base_url = (
    os.getenv("MEMORY_EMBEDDING_BASE_URL")
    or os.getenv("EMBEDDING_BASE_URL")
    or openai_base_url
)
memory_top_k_default = int(os.getenv("MEMORY_TOP_K", "5"))
memory_min_score = float(os.getenv("MEMORY_MIN_SCORE", "0.2"))
try:
    memory_summary_every = int(os.getenv("MEMORY_SUMMARY_EVERY", "5"))
except ValueError:
    memory_summary_every = 5
if memory_summary_every < 1:
    memory_summary_every = 1


def load_preferences_from_memories() -> str | None:
    path = memory_root / "preferences.txt"
    if not path.exists() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return text or None

_prompt_file_priority = [
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            if any(s in k.lower() for s in ["key", "token", "secret", "password"]):
                redacted[k] = "***"
            else:
                redacted[k] = _redact(v)
        return redacted
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class ToolLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        print(f"\n[tool start] {name}")
        try:
            parsed = json.loads(input_str)
            safe = _redact(parsed)
            print(f"[tool input] {json.dumps(safe, ensure_ascii=False)}")
        except Exception:
            print(f"[tool input] {input_str}")

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        redacted = _redact(output)
        text = str(redacted)
        if len(text) > 2000:
            text = text[:2000] + " ... (truncated)"
        print(f"[tool output] {text}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"[tool error] {type(error).__name__}: {error}")


def _load_preferences_from_store(store) -> str | None:
    """从持久化 Store 读取 /preferences.txt 的内容。"""
    try:
        item = store.get(("filesystem",), "/preferences.txt")
    except Exception:
        return None
    if item is None:
        return None
    value = getattr(item, "value", None)
    if not isinstance(value, dict):
        return None
    content = value.get("content")
    if not isinstance(content, list):
        return None
    text = "\n".join([str(line) for line in content]).strip()
    return text or None


def _format_preferences_prompt(preferences_text: str) -> str:
    return f"已知长期偏好（来自 /memories/preferences.txt）：\n{preferences_text}"


def _build_memory_embeddings() -> OpenAIEmbeddings:
    embedding_kwargs: dict[str, Any] = {
        "model": memory_embedding_model_name,
        "api_key": memory_embedding_api_key,
    }
    if memory_embedding_base_url:
        embedding_kwargs["base_url"] = memory_embedding_base_url
    return OpenAIEmbeddings(**embedding_kwargs)


class MemoryManager:
    def __init__(
        self,
        memory_dir: Path,
        db_path: Path,
        summary_model: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        max_messages: int,
        min_score: float,
        summary_every: int = 5,
        habits_root: Path | None = None,
    ):
        if sqlite_vec is None:
            raise RuntimeError(f"缺少 sqlite-vec 依赖: {_sqlite_vec_import_error}")
        self.memory_dir = Path(memory_dir)
        self.db_path = Path(db_path)
        self.sessions_path = self.memory_dir / "sessions.json"
        self.habits_root = Path(habits_root) if habits_root is not None else (self.memory_dir / "habits")
        self.summary_model = summary_model
        self.embeddings = embeddings
        self.max_messages = max(1, max_messages)
        self.min_score = min_score
        self.summary_every = max(1, int(summary_every))
        self._sessions = self._load_sessions()
        self._turn_count_by_thread: dict[str, int] = {}
        self._lock = threading.Lock()
        self.habits_root.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_db()
        self._ensure_meta_table()
        self._vec_dim = self._get_meta_dim()
        self._migrate_legacy_index()

    def _open_db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _ensure_meta_table(self) -> None:
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_meta (key TEXT PRIMARY KEY, value TEXT)"
            )

    def _get_meta_value(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM memory_meta WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        value = row["value"]
        return str(value) if value is not None else None

    def _set_meta_value(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def _get_meta_dim(self) -> int | None:
        row = self._conn.execute(
            "SELECT value FROM memory_meta WHERE key = 'embedding_dim'"
        ).fetchone()
        if not row:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def _set_meta_dim(self, dim: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('embedding_dim', ?)",
                (str(dim),),
            )

    def _ensure_vec_table(self, dim: int) -> None:
        existing = self._get_meta_dim()
        if existing and existing != dim:
            raise ValueError(f"向量维度不匹配: existing={existing}, new={dim}")
        if not existing:
            self._set_meta_dim(dim)
        with self._conn:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
                f"embedding float[{dim}] distance_metric=cosine,"
                "thread_id text,"
                "slug text,"
                "created_at text,"
                "+summary text,"
                "+file text"
                ")"
            )
        self._vec_dim = dim

    def _count_vec_rows(self) -> int:
        try:
            row = self._conn.execute("SELECT COUNT(1) AS cnt FROM memory_vec").fetchone()
        except sqlite3.Error:
            return 0
        if not row:
            return 0
        try:
            return int(row["cnt"])
        except (TypeError, ValueError):
            return 0

    def _migrate_legacy_index(self) -> None:
        legacy_path = self.memory_dir / "index.jsonl"
        if not legacy_path.exists():
            return
        if self._get_meta_value("legacy_index_migrated") == "1":
            return
        if self._count_vec_rows() > 0:
            self._set_meta_value("legacy_index_migrated", "1")
            return
        migrated = 0
        try:
            with legacy_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    embedding = entry.get("embedding")
                    if not isinstance(embedding, list) or not embedding:
                        continue
                    try:
                        dim = len(embedding)
                    except TypeError:
                        continue
                    self._ensure_vec_table(dim)
                    self._conn.execute(
                        "INSERT INTO memory_vec (embedding, thread_id, slug, created_at, summary, file) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            sqlite_vec.serialize_float32(embedding),
                            entry.get("thread_id", ""),
                            entry.get("slug", ""),
                            entry.get("created_at", ""),
                            entry.get("summary", ""),
                            entry.get("file", ""),
                        ),
                    )
                    migrated += 1
            if migrated:
                self._conn.commit()
            self._set_meta_value("legacy_index_migrated", "1")
            if migrated:
                print(f"[memory] 已迁移旧索引条目: {migrated}")
        except Exception as exc:
            print(f"[memory] 迁移旧索引失败: {exc}")

    def _load_sessions(self) -> dict[str, dict[str, str]]:
        if not self.sessions_path.exists():
            return {}
        try:
            data = json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        return {}

    def _save_sessions(self) -> None:
        self.sessions_path.write_text(
            json.dumps(self._sessions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content).strip()

    def _build_recent_text(self, messages: list[dict[str, Any]]) -> str:
        recent = messages[-self.max_messages :]
        parts: list[str] = []
        for msg in recent:
            role = str(msg.get("role", "")).strip()
            content = self._stringify_content(msg.get("content", ""))
            if not content:
                continue
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _summarize(self, recent_text: str) -> str | None:
        if not recent_text:
            return None
        prompt = (
            "请用中文总结下面对话，要求：\n"
            "1) 80-160字；\n"
            "2) 保留关键事实、偏好、待办（如果有）；\n"
            "3) 不要逐字复述。\n\n"
            f"{recent_text[:4000]}"
        )
        result = self.summary_model.invoke(prompt)
        content = getattr(result, "content", None)
        text = str(content if content is not None else result).strip()
        return text or None

    def _sanitize_slug(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug[:30]

    def _generate_slug(self, summary: str) -> str | None:
        prompt = (
            "根据以下摘要生成 1-3 个英文单词组成的文件名 slug，"
            "使用小写字母和连字符，不要扩展名，只返回 slug。\n\n"
            f"摘要：\n{summary[:1000]}"
        )
        result = self.summary_model.invoke(prompt)
        content = getattr(result, "content", None)
        raw = str(content if content is not None else result).strip().splitlines()[0].strip()
        slug = self._sanitize_slug(raw)
        return slug or None

    def _ensure_session(self, thread_id: str, summary: str, now: datetime) -> dict[str, str]:
        existing = self._sessions.get(thread_id)
        if existing:
            return existing
        slug = self._generate_slug(summary)
        if not slug:
            time_slug = now.strftime("%H%M")
            slug = time_slug
        date_str = now.date().isoformat()
        filename = f"{date_str}-{slug}.md"
        session = {"slug": slug, "file": filename}
        self._sessions[thread_id] = session
        self._save_sessions()
        return session

    def _ensure_memory_file(self, file_path: Path, thread_id: str, slug: str, now: datetime) -> None:
        if file_path.exists():
            return
        header = [
            "# 记忆",
            "",
            f"- thread_id: {thread_id}",
            f"- slug: {slug}",
            f"- created_at: {now.isoformat()}",
            "",
        ]
        file_path.write_text("\n".join(header), encoding="utf-8")

    def _append_memory_entry(self, file_path: Path, now: datetime, summary: str) -> None:
        block = [
            f"## {now.isoformat()}",
            "",
            summary,
            "",
        ]
        with file_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(block))

    def record_turn(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        try:
            with self._lock:
                turn_count = self._turn_count_by_thread.get(thread_id, 0) + 1
                self._turn_count_by_thread[thread_id] = turn_count
            if turn_count % self.summary_every != 0:
                return

            recent_text = self._build_recent_text(messages)
            summary = self._summarize(recent_text)
            if not summary:
                return
            now = datetime.now(timezone.utc)
            session = self._ensure_session(thread_id, summary, now)
            file_path = self.memory_dir / session["file"]
            embedding = self.embeddings.embed_query(summary)
            with self._lock:
                self._ensure_vec_table(len(embedding))
                self._ensure_memory_file(file_path, thread_id, session["slug"], now)
                self._append_memory_entry(file_path, now, summary)
                self._conn.execute(
                    "INSERT INTO memory_vec (embedding, thread_id, slug, created_at, summary, file) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sqlite_vec.serialize_float32(embedding),
                        thread_id,
                        session["slug"],
                        now.isoformat(),
                        summary,
                        str(file_path),
                    ),
                )
                self._conn.commit()
        except Exception as exc:
            print(f"[memory] 记忆写入失败: {exc}")

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query = str(query or "").strip()
        if not query:
            return []
        if not self._get_meta_dim():
            return []
        query_vec = self.embeddings.embed_query(query)
        results: list[dict[str, Any]] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, distance, thread_id, slug, created_at, summary, file "
                "FROM memory_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(query_vec), max(1, int(top_k))),
            ).fetchall()
        for row in rows:
            distance = float(row["distance"])
            score = max(0.0, 1.0 - distance)
            if score < self.min_score:
                continue
            results.append(
                {
                    "score": round(score, 4),
                    "distance": round(distance, 4),
                    "created_at": row["created_at"],
                    "summary": row["summary"],
                    "file": row["file"],
                    "thread_id": row["thread_id"],
                    "slug": row["slug"],
                }
            )
        return results



def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def _iter_prompt_files() -> list[Path]:
    """按优先级列出 prompts 目录中的提示词文件。"""
    if not prompts_dir.exists():
        return []

    selected: list[Path] = []
    seen_names: set[str] = set()

    for filename in _prompt_file_priority:
        path = prompts_dir / filename
        if path.exists() and path.is_file():
            selected.append(path)
            seen_names.add(path.name.lower())

    extras = sorted(
        [
            p
            for p in prompts_dir.glob("*.md")
            if p.is_file() and p.name.lower() not in seen_names
        ],
        key=lambda p: p.name.lower(),
    )
    selected.extend(extras)
    return selected


def load_prompts() -> str:
    """读取 prompts 目录并拼接提示词内容。"""
    files = _iter_prompt_files()
    if not files:
        return ""

    contents: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                contents.append(text)
        except Exception as exc:
            print(f"[prompts] 读取失败: {path.name} {exc}")
    return "\n\n---\n\n".join(contents)


_base_system_prompt = """You are a helpful assistant.
If the user shares preferences, save them to /memories/preferences.txt so you can
remember them in future conversations.
For complex tasks, break down steps and use the write_todos tool to track progress.
If you need long-term memory, use the memory_search tool.
If browsing websites or interacting with web pages is requested, use the browser tool.
"""

_loaded_prompts = load_prompts()
if _loaded_prompts:
    system_prompt = _loaded_prompts + "\n\n---\n\n" + _base_system_prompt
    loaded_files = _iter_prompt_files()
    print(f"[prompts] 已加载 {len(loaded_files)} 个提示词文件，总长度 {len(_loaded_prompts)} 字节")
else:
    system_prompt = _base_system_prompt

chat_model = ChatOpenAI(
    model=openai_model,
    api_key=openai_api_key,
    base_url=openai_base_url,
)

summary_model_kwargs = {
    "model": memory_summary_model_name,
    "api_key": memory_summary_api_key,
}
if openai_base_url:
    summary_model_kwargs["base_url"] = openai_base_url
memory_summary_model = ChatOpenAI(**summary_model_kwargs)

memory_embeddings = _build_memory_embeddings()
print(f"[memory] embedding model={memory_embedding_model_name}")

memory_manager = MemoryManager(
    memory_dir=memory_root,
    db_path=memory_db_path,
    summary_model=memory_summary_model,
    embeddings=memory_embeddings,
    max_messages=memory_max_messages,
    min_score=memory_min_score,
    summary_every=memory_summary_every,
)


@lc_tool
def memory_search(query: str, top_k: int = memory_top_k_default) -> str:
    """向量检索本地记忆，返回匹配摘要。"""
    try:
        results = memory_manager.search(query=query, top_k=top_k)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: {exc}"




checkpointer = MemorySaver()


class LocalExecBackend(FilesystemBackend, SandboxBackendProtocol):
    def __init__(self, root_dir: str, backend_id: str, virtual_mode: bool = True):
        super().__init__(root_dir=root_dir, virtual_mode=virtual_mode)
        os.makedirs(self.cwd, exist_ok=True)
        self._id = f"{backend_id}:{self.cwd}"

    @property
    def id(self) -> str:
        return self._id

    # 在配置的根目录内执行命令。
    def execute(self, command: str) -> ExecuteResponse:
        try:
            proc = subprocess.run(  # noqa: S603
                command,
                shell=True,
                cwd=str(self.cwd),
                capture_output=True,
                text=True,
                encoding=exec_output_encoding,
                errors="replace",
                timeout=sandbox_timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            truncated = False
            if len(output) > 20000:
                output = output[:20000] + "\n... [truncated]"
                truncated = True
            return ExecuteResponse(output=output, exit_code=proc.returncode, truncated=truncated)
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Command timed out after {sandbox_timeout}s.",
                exit_code=124,
                truncated=False,
            )


class SkillsBackend(FilesystemBackend):
    @staticmethod
    def _norm(path: str) -> str:
        normalized = path.replace("\\", "/")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        return normalized

    def ls_info(self, path: str) -> list[dict]:
        infos = super().ls_info(self._norm(path))
        for item in infos:
            if "path" in item:
                item["path"] = self._norm(item["path"])
        return infos

    def download_files(self, paths: list[str]):
        norm_paths = [self._norm(p) for p in paths]
        return super().download_files(norm_paths)


class HistoryBackend(FilesystemBackend):
    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        # 先尝试标准编辑；失败后再做换行符兼容处理。
        result = super().edit(file_path, old_string, new_string, replace_all=replace_all)
        if result.error is None or "String not found in file:" not in result.error:
            return result

        resolved_path = self._resolve_path(file_path)
        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")
        try:
            content = resolved_path.read_text(encoding="utf-8")
            norm_content = self._normalize_newlines(content)
            norm_old = self._normalize_newlines(old_string)
            norm_new = self._normalize_newlines(new_string)

            # 1) 先尝试在归一化后的内容中做替换
            replacement = perform_string_replacement(norm_content, norm_old, norm_new, replace_all)
            if not isinstance(replacement, str):
                new_content, occurrences = replacement
                resolved_path.write_text(new_content, encoding="utf-8")
                return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))

            # 2) 若新内容已在文件中，直接视为成功（避免重复写入）
            if norm_new in norm_content:
                return EditResult(path=file_path, files_update=None, occurrences=1)

            # 3) 兜底：将“新增部分”直接追加到文件末尾
            if norm_new.startswith(norm_old):
                tail = norm_new[len(norm_old) :]
            else:
                tail = norm_new
            resolved_path.write_text(norm_content + tail, encoding="utf-8")
            return EditResult(path=file_path, files_update=None, occurrences=1)
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")


skills_root = Path(skills_dir).resolve()
skills_root.mkdir(parents=True, exist_ok=True)
skills_available = any(skills_root.rglob("SKILL.md"))
supports_skills = "skills" in inspect.signature(create_deep_agent).parameters
if skills_available and not supports_skills:
    print("提示：当前 deepagents 版本不支持 skills 参数，已自动忽略 skills 配置。")
skills_arg = ["/skills/"] if (skills_available and supports_skills) else []
skills_backend = SkillsBackend(root_dir=str(skills_root), virtual_mode=True) if skills_available else None
history_backend = HistoryBackend(root_dir=history_root, virtual_mode=True)
memories_backend = FilesystemBackend(root_dir=str(memory_root), virtual_mode=True)


def _load_skills_metadata() -> list[dict[str, Any]]:
    if skills_backend is None:
        return []
    try:
        from deepagents.middleware import skills as skills_mod
    except Exception:
        return []

    # 使用与运行时一致的路由，避免路径在 Windows 上失真。
    default_backend = FilesystemBackend(root_dir=local_shell_root, virtual_mode=False)
    backend = CompositeBackend(default=default_backend, routes={"/skills/": skills_backend})
    try:
        return skills_mod._list_skills(backend, "/skills/")
    except Exception as exc:
        if log_tool_calls:
            print(f"技能加载失败: {exc}")
        return []


def _format_skills_overview(skills_metadata: list[dict[str, Any]]) -> str:
    if not skills_metadata:
        return ""
    lines = [
        "以下是当前可用的 Skills（技能），当用户需求匹配时优先阅读对应 SKILL.md：",
    ]
    for skill in skills_metadata:
        name = str(skill.get("name", "")).strip()
        desc = str(skill.get("description", "")).strip()
        path = str(skill.get("path", "")).strip()
        if not name:
            continue
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
        if path:
            lines.append(f"  说明文件: {path}")
    return "\n".join(lines)


def make_backend(runtime):
    # 已移除沙盒机制：默认使用本地执行与本地文件系统。
    exec_backend = LocalExecBackend(
        root_dir=local_shell_root,
        backend_id="local-shell",
        virtual_mode=False,
    )
    routes = {
        "/memories/": memories_backend,
        # 对会话历史文件追加做换行兼容，避免 Windows CRLF 造成 edit 失败。
        "/conversation_history/": history_backend,
        # 将执行根目录映射为 /workspace/，供文件工具访问。
        "/workspace/": exec_backend,
    }
    if skills_backend is not None:
        routes["/skills/"] = skills_backend
    return CompositeBackend(
        default=exec_backend,
        routes=routes,
    )


 


def _normalize_mcp_tools(config: dict) -> dict:
    tools = config.get("tools", config)
    normalized = {}
    for name, entry in tools.items():
        if isinstance(entry, dict) and entry.get("active") is False:
            continue
        if not isinstance(entry, dict):
            continue
        server = {}
        if "config" in entry and isinstance(entry["config"], dict):
            server.update(entry["config"])
        else:
            server.update(entry)
        if "type" in entry and "transport" not in server:
            server["transport"] = entry["type"]
        normalized[name] = server
    return normalized or None


def _command_exists(command: str) -> bool:
    cmd = str(command or "").strip()
    if not cmd:
        return False
    expanded = os.path.expandvars(os.path.expanduser(cmd))
    is_path_like = (
        expanded.startswith(".")
        or expanded.startswith("/")
        or expanded.startswith("\\")
        or (len(expanded) > 1 and expanded[1] == ":")
        or "/" in expanded
        or "\\" in expanded
    )
    if is_path_like:
        return os.path.isfile(expanded) or shutil.which(expanded) is not None
    return shutil.which(expanded) is not None


def _filter_unavailable_mcp_servers(servers: dict | None) -> dict | None:
    if not servers:
        return None
    filtered: dict[str, dict] = {}
    for name, server in servers.items():
        if not isinstance(server, dict):
            continue
        transport = str(server.get("transport", "stdio")).strip().lower()
        if transport == "stdio":
            command = str(server.get("command", "")).strip()
            if not _command_exists(command):
                print(
                    f"Skip MCP server '{name}': command not found: {command or '(empty)'}"
                )
                continue
        filtered[name] = server
    return filtered or None


def load_mcp_servers():
    raw = os.getenv("MCP_SERVERS")
    if raw:
        try:
            return _filter_unavailable_mcp_servers(_normalize_mcp_tools(json.loads(raw)))
        except json.JSONDecodeError:
            print("MCP_SERVERS is not valid JSON. Skipping MCP tools.")
            return None
    config_path = os.getenv("MCP_CONFIG", "mcp.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return _filter_unavailable_mcp_servers(
                    _normalize_mcp_tools(json.load(f))
                )
        except json.JSONDecodeError:
            print("mcp.json is not valid JSON. Skipping MCP tools.")
            return None
    return None


async def build_agent(extra_tools: list[Any] | None = None):
    tools = [internet_search, memory_search]
    if extra_tools:
        tools.extend(extra_tools)
    if get_browser_tools is not None:
        try:
            tools.extend(get_browser_tools())
        except Exception as exc:
            print(f"browser tools load failed: {exc}")
    tools.extend(get_gemini_image_tools())
    tools.extend(get_tts_tools())
    mcp_servers = load_mcp_servers()
    if mcp_servers and MultiServerMCPClient:
        try:
            mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = await mcp_client.get_tools()
            tools.extend(mcp_tools)
        except Exception as exc:
            print(f"MCP tools load failed, continue without MCP: {exc}")
    elif mcp_servers and not MultiServerMCPClient:
        print("langchain-mcp-adapters is not installed. MCP tools disabled.")

    skills_metadata = _load_skills_metadata()
    skills_overview = _format_skills_overview(skills_metadata)
    runtime_system_prompt = system_prompt
    if skills_overview:
        runtime_system_prompt = runtime_system_prompt + "\n\n" + skills_overview
        if log_tool_calls:
            skill_names = [s.get("name", "") for s in skills_metadata if s.get("name")]
            if skill_names:
                print("已加载 Skills:", ", ".join(skill_names))

    extra_middleware = [
        ModelRetryMiddleware(max_retries=2, backoff_factor=1.5, initial_delay=0.5),
        ToolRetryMiddleware(max_retries=2, backoff_factor=1.5),
    ]

    agent_kwargs = dict(
        model=chat_model,
        tools=tools,
        system_prompt=runtime_system_prompt,
        middleware=extra_middleware,
        backend=make_backend,
        checkpointer=checkpointer,
    )
    if supports_skills and skills_arg:
        agent_kwargs["skills"] = skills_arg
    agent = create_deep_agent(**agent_kwargs)
    tool_names = []
    for t in tools:
        if hasattr(t, "name"):
            tool_names.append(t.name)
        elif hasattr(t, "__name__"):
            tool_names.append(t.__name__)
        elif isinstance(t, dict) and "name" in t:
            tool_names.append(str(t["name"]))
        else:
            tool_names.append(str(t))
    print("Tools loaded:", ", ".join(tool_names))
    return agent


async def main():
    agent = await build_agent()
    thread_id = os.getenv("DEEPAGENT_THREAD_ID", str(uuid.uuid4()))
    session_messages_map: dict[str, list[dict[str, Any]]] = {thread_id: []}
    print(f"DeepAgent demo started. thread_id={thread_id}")
    print("Type /new to start a new thread, /exit to quit.")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nYou: ")
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            break
        if user_input.lower() == "/new":
            thread_id = str(uuid.uuid4())
            session_messages_map[thread_id] = []
            print(f"New thread_id={thread_id}")
            continue

        session_messages = session_messages_map.setdefault(thread_id, [])
        session_messages.append({"role": "user", "content": user_input})
        config = {"configurable": {"thread_id": thread_id}}
        if log_tool_calls:
            config["callbacks"] = [ToolLogger()]

        preferences_text = load_preferences_from_memories()
        messages: list[dict[str, Any]] = []
        if preferences_text:
            messages.append(
                {
                    "role": "system",
                    "content": _format_preferences_prompt(preferences_text),
                }
            )
        messages.append({"role": "user", "content": user_input})
        payload = {"messages": messages}

        result = await agent.ainvoke(
            payload,
            config=config,
        )
        reply = result["messages"][-1].content
        reply_text = str(reply)
        print(f"Assistant: {reply_text}")
        session_messages.append({"role": "assistant", "content": reply_text})
        await asyncio.to_thread(memory_manager.record_turn, thread_id, session_messages)


if __name__ == "__main__":
    asyncio.run(main())
