"""Browser automation module for DeepAgent.

Provides browser control capabilities using DrissionPage (CDP-based).
"""

from .session import BrowserSession, get_session
from .browser_tool import browser, get_browser_tools

__all__ = [
    "BrowserSession",
    "get_session",
    "browser",
    "get_browser_tools",
]
