"""基于 DrissionPage 的浏览器工具。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .config import get_config
from .locator import normalize_selectors, resolve_element
from .session import get_session
from .snapshot import get_snapshot


def _ensure_screenshot_dir() -> Path:
    """确保截图目录存在。"""
    config = get_config()
    path = Path(config.screenshot_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_seconds(ms: int | None, default_ms: int) -> float:
    """毫秒转秒，最小保留 0.1 秒。"""
    value = default_ms if ms is None or ms <= 0 else ms
    return max(value / 1000.0, 0.1)


def _safe_json_value(value: Any) -> Any:
    """把对象尽量转换成可 JSON 序列化的值。"""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)


def _json_result(data: dict[str, Any]) -> str:
    """统一 JSON 输出。"""
    return json.dumps(data, ensure_ascii=False)


@tool
def browser(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    headless: bool = False,
    format: str = "interactive",
    path: str = "",
    full_page: bool = True,
    script: str = "",
    time_ms: int = 1000,
    direction: str = "down",
    amount: int = 500,
    by_js: bool = False,
    clear: bool = False,
    locator_timeout_ms: int = 0,
    index: int = 1,
    strict: bool = False,
    search_frames: bool = True,
) -> str:
    """浏览器自动化入口。

    常用动作：
    - `start` / `stop` / `status`
    - `navigate` / `back` / `forward` / `refresh`
    - `click` / `type` / `hover`
    - `wait` / `scroll`
    - `snapshot` / `screenshot`
    - `evaluate` / `alert`

    定位增强：
    - `selector` 支持 `||` 或换行写多个候选定位符。
    - 支持自动识别 CSS/XPath 写法。
    - `index` 为 1 基索引，负数表示倒序。
    - `strict=True` 时不做智能优选，按原始匹配顺序取元素。
    """
    session = get_session()
    config = get_config()

    locator_timeout_sec = _to_seconds(locator_timeout_ms, config.default_timeout_ms)

    try:
        if action == "start":
            result = session.start(headless=headless)
            return _json_result(result)

        if action == "stop":
            result = session.stop()
            return _json_result(result)

        if action == "status":
            result = session.get_info()
            return _json_result(result)

        if action == "navigate":
            if not url:
                return _json_result({"status": "error", "message": "url is required"})
            result = session.navigate(url)
            return _json_result(result)

        page = session.page

        if action == "back":
            page.back()
            return _json_result({"status": "ok", "action": "back", "url": page.url})

        if action == "forward":
            page.forward()
            return _json_result({"status": "ok", "action": "forward", "url": page.url})

        if action == "refresh":
            page.refresh()
            return _json_result({"status": "ok", "action": "refresh", "url": page.url})

        if action in {"click", "hover", "type"}:
            if not selector:
                return _json_result({"status": "error", "message": "selector is required"})

            prefer = "click" if action == "click" else "input" if action == "type" else "visible"
            loc_result = resolve_element(
                page,
                selector,
                timeout=locator_timeout_sec,
                index=index,
                prefer=prefer,
                strict=strict,
                search_frames=search_frames,
            )
            if not loc_result.found:
                return _json_result(
                    {
                        "status": "error",
                        "message": loc_result.error,
                        "selector": selector,
                        "tried": loc_result.tried_selectors,
                    }
                )

            ele = loc_result.element

            if action == "click":
                clicked = ele.click(by_js=by_js, timeout=locator_timeout_sec)
                real_by_js = by_js
                if clicked is False and not by_js:
                    # 模拟点击失败后自动退回 JS 点击，提升执行成功率
                    ele.click(by_js=True)
                    real_by_js = True

                return _json_result(
                    {
                        "status": "ok",
                        "action": "clicked",
                        "selector": selector,
                        "used_selector": loc_result.used_selector,
                        "source": loc_result.source,
                        "matched_count": loc_result.matched_count,
                        "by_js": real_by_js,
                    }
                )

            if action == "hover":
                page.scroll.to_see(ele)
                ele.hover()
                return _json_result(
                    {
                        "status": "ok",
                        "action": "hovered",
                        "selector": selector,
                        "used_selector": loc_result.used_selector,
                        "source": loc_result.source,
                        "matched_count": loc_result.matched_count,
                    }
                )

            if not text:
                return _json_result({"status": "error", "message": "text is required"})

            ele.input(text, clear=clear)
            return _json_result(
                {
                    "status": "ok",
                    "action": "typed",
                    "selector": selector,
                    "used_selector": loc_result.used_selector,
                    "source": loc_result.source,
                    "matched_count": loc_result.matched_count,
                    "text_length": len(text),
                    "cleared": clear,
                }
            )

        if action == "screenshot":
            if path:
                file_path = Path(path).resolve()
                save_path = str(file_path.parent)
                save_name = file_path.name
            else:
                screenshot_dir = _ensure_screenshot_dir()
                save_path = str(screenshot_dir)
                save_name = f"screenshot_{uuid.uuid4().hex[:8]}.jpg"

            Path(save_path).mkdir(parents=True, exist_ok=True)
            result_path = page.get_screenshot(
                path=save_path,
                name=save_name,
                full_page=full_page,
                as_bytes=None,
                as_base64=None,
            )
            return _json_result({"status": "ok", "path": result_path, "full_page": full_page})

        if action == "snapshot":
            snapshot_text = get_snapshot(page, format=format, max_chars=config.snapshot_max_chars)
            return _json_result({"status": "ok", "format": format, "content": snapshot_text})

        if action == "wait":
            if selector:
                timeout_sec = _to_seconds(time_ms, config.default_timeout_ms)
                candidates = normalize_selectors(selector)
                if not candidates:
                    return _json_result({"status": "error", "message": "selector is empty"})

                if len(candidates) == 1:
                    loaded = page.wait.eles_loaded(candidates[0], timeout=timeout_sec, any_one=True, raise_err=False)
                else:
                    loaded = page.wait.eles_loaded(candidates, timeout=timeout_sec, any_one=True, raise_err=False)

                if not loaded:
                    return _json_result(
                        {
                            "status": "timeout",
                            "action": "wait_element",
                            "selector": selector,
                            "normalized": candidates,
                        }
                    )

                loc_result = resolve_element(
                    page,
                    selector,
                    timeout=0.5,
                    index=index,
                    prefer="visible",
                    strict=strict,
                    search_frames=search_frames,
                )
                return _json_result(
                    {
                        "status": "ok" if loc_result.found else "timeout",
                        "action": "wait_element",
                        "selector": selector,
                        "normalized": candidates,
                        "used_selector": loc_result.used_selector,
                        "source": loc_result.source,
                        "matched_count": loc_result.matched_count,
                        "tried": loc_result.tried_selectors,
                    }
                )

            page.wait(_to_seconds(time_ms, 1000))
            return _json_result({"status": "ok", "action": "wait_time", "time_ms": time_ms})

        if action == "scroll":
            if direction == "up":
                page.scroll.up(amount)
            elif direction == "down":
                page.scroll.down(amount)
            elif direction == "left":
                page.scroll.left(amount)
            elif direction == "right":
                page.scroll.right(amount)
            elif direction == "top":
                page.scroll.to_top()
            elif direction == "bottom":
                page.scroll.to_bottom()
            elif direction == "leftmost":
                page.scroll.to_leftmost()
            elif direction == "rightmost":
                page.scroll.to_rightmost()
            else:
                page.scroll.down(amount)

            return _json_result(
                {
                    "status": "ok",
                    "action": "scrolled",
                    "direction": direction,
                    "amount": amount,
                }
            )

        if action == "evaluate":
            if not script:
                return _json_result({"status": "error", "message": "script is required"})

            result = page.run_js(script)
            return _json_result({"status": "ok", "result": _safe_json_value(result)})

        if action == "alert":
            timeout_sec = _to_seconds(time_ms, 5000)
            result = page.handle_alert(accept=True, send=text if text else None, timeout=timeout_sec)
            return _json_result(
                {
                    "status": "ok",
                    "action": "alert_handled",
                    "text": result if result else None,
                }
            )

        return _json_result(
            {
                "status": "error",
                "message": f"Unknown action: {action}",
                "available_actions": [
                    "start",
                    "stop",
                    "status",
                    "navigate",
                    "back",
                    "forward",
                    "refresh",
                    "click",
                    "hover",
                    "type",
                    "screenshot",
                    "snapshot",
                    "wait",
                    "scroll",
                    "evaluate",
                    "alert",
                ],
            }
        )

    except RuntimeError as e:
        return _json_result(
            {
                "status": "error",
                "message": str(e),
                "hint": "请先调用 browser(action='start')。",
            }
        )
    except Exception as e:
        return _json_result({"status": "error", "message": str(e)})


def get_browser_tools() -> list:
    """返回工具列表。"""
    return [browser]
