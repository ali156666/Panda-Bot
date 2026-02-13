"""浏览器会话管理。"""
from __future__ import annotations

import threading
from typing import Any, Optional

from .config import BrowserConfig, get_config


class BrowserSession:
    """管理浏览器生命周期，提供全局单例会话。"""

    def __init__(self, config: Optional[BrowserConfig] = None):
        self._config = config or get_config()
        self._page = None
        self._browser = None
        self._lock = threading.Lock()

    @property
    def page(self):
        """返回当前页面对象。"""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    def is_active(self) -> bool:
        """浏览器是否已启动。"""
        return self._page is not None

    def _apply_runtime_settings(self) -> None:
        """应用运行时设置，提升定位和交互稳定性。"""
        if self._page is None:
            return

        try:
            base_timeout = max(self._config.default_timeout_ms / 1000, 0.1)
            page_load_timeout = max(self._config.navigation_timeout_ms / 1000, 0.1)
            self._page.set.timeouts(base=base_timeout, page_load=page_load_timeout)
        except Exception:
            pass

        try:
            # 按官方建议，关闭平滑滚动可降低点击不准问题
            self._page.set.scroll.smooth(on_off=False)
            self._page.set.scroll.wait_complete(on_off=True)
        except Exception:
            pass

    def start(self, headless: Optional[bool] = None) -> dict[str, Any]:
        """启动浏览器。"""
        with self._lock:
            if self._page is not None:
                return {"status": "already_running", "message": "Browser is already running."}

            try:
                from DrissionPage import Chromium, ChromiumOptions
            except ImportError:
                return {
                    "status": "error",
                    "message": "DrissionPage not installed. Run: pip install DrissionPage",
                }

            try:
                co = ChromiumOptions()
                co.auto_port()

                use_headless = headless if headless is not None else self._config.headless
                if use_headless:
                    co.headless(True)
                    co.set_argument("--headless=new")

                if self._config.no_sandbox:
                    co.set_argument("--no-sandbox")
                    co.set_argument("--disable-setuid-sandbox")

                if self._config.disable_gpu:
                    co.set_argument("--disable-gpu")

                if self._config.proxy:
                    co.set_proxy(self._config.proxy)

                co.set_argument("--disable-dev-shm-usage")
                co.set_argument("--no-first-run")
                co.set_argument("--disable-background-networking")

                self._browser = Chromium(co)
                self._page = self._browser.latest_tab
                self._apply_runtime_settings()

                return {
                    "status": "started",
                    "message": "Browser started successfully.",
                    "headless": use_headless,
                }
            except Exception as e:
                return {"status": "error", "message": f"Failed to start browser: {str(e)}"}

    def stop(self) -> dict[str, Any]:
        """关闭浏览器。"""
        with self._lock:
            if self._browser is None and self._page is None:
                return {"status": "not_running", "message": "Browser is not running."}

            try:
                if self._browser is not None:
                    self._browser.quit()
                elif self._page is not None:
                    self._page.quit()

                self._browser = None
                self._page = None
                return {"status": "stopped", "message": "Browser stopped successfully."}
            except Exception as e:
                self._browser = None
                self._page = None
                return {"status": "error", "message": f"Error stopping browser: {str(e)}"}

    def navigate(self, url: str) -> dict[str, Any]:
        """访问页面。"""
        if not self.is_active():
            return {"status": "error", "message": "Browser not started."}

        try:
            timeout_sec = max(self._config.navigation_timeout_ms / 1000, 0.1)
            self._page.get(url, timeout=timeout_sec)
            try:
                self._page.wait.doc_loaded(timeout=timeout_sec, raise_err=False)
            except Exception:
                pass

            return {
                "status": "ok",
                "url": self._page.url,
                "title": self._page.title,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_info(self) -> dict[str, Any]:
        """获取当前页面信息。"""
        if not self.is_active():
            return {"status": "not_running"}

        try:
            return {
                "status": "running",
                "url": self._page.url,
                "title": self._page.title,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


_session: Optional[BrowserSession] = None
_session_lock = threading.Lock()


def get_session() -> BrowserSession:
    """获取全局会话单例。"""
    global _session
    with _session_lock:
        if _session is None:
            _session = BrowserSession()
        return _session


def reset_session() -> None:
    """重置全局会话（测试用）。"""
    global _session
    with _session_lock:
        if _session is not None:
            try:
                _session.stop()
            except Exception:
                pass
            _session = None
