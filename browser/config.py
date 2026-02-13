"""Browser configuration settings."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""
    
    # Browser settings
    headless: bool = False
    no_sandbox: bool = True
    disable_gpu: bool = False
    
    # Timeouts (ms)
    default_timeout_ms: int = 30000
    navigation_timeout_ms: int = 60000
    
    # Screenshot settings
    screenshot_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "..", "outputs", "screenshots"
    ))
    
    # Snapshot settings
    snapshot_max_chars: int = 50000
    
    # Proxy
    proxy: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "BrowserConfig":
        """Load config from environment variables."""
        return cls(
            headless=os.getenv("BROWSER_HEADLESS", "false").lower() == "true",
            no_sandbox=os.getenv("BROWSER_NO_SANDBOX", "true").lower() == "true",
            proxy=os.getenv("BROWSER_PROXY"),
        )


# Global config instance
_config: Optional[BrowserConfig] = None


def get_config() -> BrowserConfig:
    """Get or create browser config."""
    global _config
    if _config is None:
        _config = BrowserConfig.from_env()
    return _config
