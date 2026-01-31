import asyncio
import json
import os
import sys
import threading
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
)

import deepagent_demo as runtime


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"缺少环境变量：{name}")
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


async def _handle_message(
    data: P2ImMessageReceiveV1,
    agent,
    client: lark.Client,
) -> None:
    message = data.event.message
    sender_open_id = _safe_getattr(data, "event.sender.sender_id.open_id")
    sender_user_id = _safe_getattr(data, "event.sender.sender_id.user_id")
    print(
        "[feishu] 收到消息"
        f" chat_id={message.chat_id}"
        f" message_id={message.message_id}"
        f" chat_type={message.chat_type}"
        f" msg_type={message.message_type}"
        f" sender_open_id={sender_open_id}"
        f" sender_user_id={sender_user_id}"
    )
    text = _extract_text_message(data)
    if not text:
        print("[feishu] 忽略非单聊或非文本消息。")
        return

    thread_id = data.event.message.chat_id
    config = {"configurable": {"thread_id": thread_id}}
    if runtime.log_tool_calls:
        config["callbacks"] = [runtime.ToolLogger()]

    payload = {"messages": [{"role": "user", "content": text}]}
    print(f"[feishu] 用户问题: {_truncate_text(text)}")
    result = await agent.ainvoke(payload, config=config)
    reply = result["messages"][-1].content
    reply_text = reply if isinstance(reply, str) else str(reply)
    print(f"[feishu] AI 回复: {_truncate_text(reply_text)}")

    content = json.dumps({"text": reply_text}, ensure_ascii=False)
    request = (
        CreateMessageRequest.builder()
        .receive_id_type("chat_id")
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(data.event.message.chat_id)
            .msg_type("text")
            .content(content)
            .build()
        )
        .build()
    )
    response = client.im.v1.message.create(request)
    if not response.success():
        raise RuntimeError(
            "发送消息失败，code={code} msg={msg} log_id={log_id}".format(
                code=response.code,
                msg=response.msg,
                log_id=response.get_log_id(),
            )
        )


def _start_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _build_agent():
    if runtime.PostgresStore is None:
        print("PostgresStore 不可用，请安装 langgraph-checkpoint-postgres 和 psycopg。")
        sys.exit(1)
    db_uri = runtime.build_db_uri()
    if not db_uri:
        print("缺少 Postgres 配置，请设置 DATABASE_URL 或 PG_* 环境变量。")
        sys.exit(1)
    store_ctx = runtime.PostgresStore.from_conn_string(db_uri)
    store = store_ctx.__enter__()
    agent = await runtime.build_agent(store)
    return agent, store_ctx


def main() -> None:
    app_id, app_secret = _setup_lark_env()
    agent, store_ctx = asyncio.run(_build_agent())

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=_start_loop, args=(loop,), daemon=True)
    loop_thread.start()

    client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
    if runtime.log_tool_calls:
        print("[feishu] 已开启工具调用日志输出。")

    def _on_message(data: P2ImMessageReceiveV1) -> None:
        future = asyncio.run_coroutine_threadsafe(_handle_message(data, agent, client), loop)

        def _log_result(task: asyncio.Future) -> None:
            try:
                task.result()
            except Exception as exc:  # pragma: no cover - 运行时日志
                print(f"处理消息失败：{exc}")

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

    try:
        print("飞书长连接已启动，等待消息中。")
        ws_client.start()
    except KeyboardInterrupt:
        print("已退出。")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        store_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
