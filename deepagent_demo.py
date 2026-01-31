import asyncio
import json
import os
import subprocess
import sys
import uuid
import inspect
from typing import Any, Literal
from pathlib import Path
from urllib.parse import quote

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.filesystem import EditResult, FilesystemBackend, perform_string_replacement
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    TodoListMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:  # pragma: no cover - optional dependency
    MultiServerMCPClient = None

try:
    from langgraph.store.postgres import PostgresStore
except Exception:  # pragma: no cover - optional dependency
    PostgresStore = None


def load_env_file(path: str = ".env") -> None:
    """Minimal .env loader; does not overwrite existing env vars."""
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
            if key and key not in os.environ:
                os.environ[key] = value


def require_env(names: list[str]) -> dict[str, str]:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        print("Missing required environment variables:", ", ".join(missing))
        print("Tip: create a .env file next to this script or set them in your shell.")
        sys.exit(1)
    return {n: os.environ[n] for n in names}


def build_db_uri() -> str | None:
    direct = os.getenv("DATABASE_URL") or os.getenv("PG_CONN_STRING")
    if direct:
        return direct
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    database = os.getenv("PG_DATABASE")
    if not all([host, user, password, database]):
        return None
    user_enc = quote(user)
    pass_enc = quote(password)
    db_enc = quote(database)
    sslmode = os.getenv("PG_SSLMODE")
    query = f"?sslmode={sslmode}" if sslmode else ""
    return f"postgresql://{user_enc}:{pass_enc}@{host}:{port}/{db_enc}{query}"


load_env_file()

env = require_env(["OPENAI_API_KEY", "TAVILY_API_KEY"])
openai_api_key = env["OPENAI_API_KEY"]
tavily_api_key = env["TAVILY_API_KEY"]

openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

tavily_client = TavilyClient(api_key=tavily_api_key)
log_tool_calls = os.getenv("LOG_TOOL_CALLS", "1") != "0"
sandbox_root = os.getenv("SANDBOX_ROOT", ".sandbox")
sandbox_timeout = int(os.getenv("SANDBOX_TIMEOUT", "20"))
os.makedirs(sandbox_root, exist_ok=True)
# 只有显式开启时，才允许 AI 在沙箱外执行命令。
allow_local_shell = os.getenv("ALLOW_LOCAL_SHELL", "0") == "1"
# 启用本地终端执行时的根目录。
local_shell_root = os.getenv("LOCAL_SHELL_ROOT", ".")
sandbox_root = os.getenv("SANDBOX_ROOT", ".sandbox")
sandbox_timeout = int(os.getenv("SANDBOX_TIMEOUT", "20"))
skills_dir = os.getenv("SKILLS_DIR", "skills")
history_root = os.getenv("HISTORY_ROOT", ".deepagents_fs")
os.makedirs(history_root, exist_ok=True)


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


system_prompt = """You are a helpful assistant.
If the user shares preferences, save them to /memories/preferences.txt so you can
remember them in future conversations.
For complex tasks, break down steps and use the write_todos tool to track progress.
"""

chat_model = ChatOpenAI(
    model=openai_model,
    api_key=openai_api_key,
    base_url=openai_base_url,
)

checkpointer = MemorySaver()


class LocalExecBackend(FilesystemBackend, SandboxBackendProtocol):
    def __init__(self, root_dir: str, backend_id: str):
        super().__init__(root_dir=root_dir, virtual_mode=True)
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
        return path.replace("\\", "/")

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


def make_backend(runtime):
    # 根据安全开关切换执行根目录。
    if allow_local_shell:
        exec_backend = LocalExecBackend(root_dir=local_shell_root, backend_id="local-shell")
    else:
        exec_backend = LocalExecBackend(root_dir=sandbox_root, backend_id="local-sandbox")
    routes = {
        "/memories/": StoreBackend(runtime),
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


def load_mcp_servers():
    raw = os.getenv("MCP_SERVERS")
    if raw:
        try:
            return _normalize_mcp_tools(json.loads(raw))
        except json.JSONDecodeError:
            print("MCP_SERVERS is not valid JSON. Skipping MCP tools.")
            return None
    config_path = os.getenv("MCP_CONFIG", "mcp.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return _normalize_mcp_tools(json.load(f))
        except json.JSONDecodeError:
            print("mcp.json is not valid JSON. Skipping MCP tools.")
            return None
    return None


async def build_agent(store):
    tools = [internet_search]
    mcp_servers = load_mcp_servers()
    if mcp_servers and MultiServerMCPClient:
        mcp_client = MultiServerMCPClient(mcp_servers)
        mcp_tools = await mcp_client.get_tools()
        tools.extend(mcp_tools)
    elif mcp_servers and not MultiServerMCPClient:
        print("langchain-mcp-adapters is not installed. MCP tools disabled.")

    extra_middleware = [
        ModelRetryMiddleware(max_retries=2, backoff_factor=1.5, initial_delay=0.5),
        ToolRetryMiddleware(max_retries=2, backoff_factor=1.5),
    ]

    agent_kwargs = dict(
        model=chat_model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=extra_middleware,
        store=store,
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
    if PostgresStore is None:
        print("PostgresStore not available. Install langgraph-checkpoint-postgres and psycopg.")
        return
    db_uri = build_db_uri()
    if not db_uri:
        print("Missing Postgres configuration. Set DATABASE_URL or PG_* env vars.")
        return
    store_ctx = PostgresStore.from_conn_string(db_uri)
    store = store_ctx.__enter__()
    try:
        store.setup()
        agent = await build_agent(store)
        thread_id = os.getenv("DEEPAGENT_THREAD_ID", str(uuid.uuid4()))
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
                print(f"New thread_id={thread_id}")
                continue

            config = {"configurable": {"thread_id": thread_id}}
            if log_tool_calls:
                config["callbacks"] = [ToolLogger()]

            payload = {"messages": [{"role": "user", "content": user_input}]}

            result = await agent.ainvoke(
                payload,
                config=config,
            )
            reply = result["messages"][-1].content
            print(f"Assistant: {reply}")
    finally:
        store_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
