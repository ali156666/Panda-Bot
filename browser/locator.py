"""浏览器定位工具。

实现要点：
1. 自动标准化常见定位输入，减少误把 CSS 当文本匹配的问题。
2. 支持一条指令里传多个候选定位符，按优先级依次尝试。
3. 对候选元素做可点击/可输入优选，降低“命中但不可操作”的概率。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


_CSS_SIGNAL_RE = re.compile(
    r"(?:^|[\s>+~])(?:[a-zA-Z][a-zA-Z0-9_-]*[#.][a-zA-Z0-9_-]+|\*)"
)


@dataclass
class LocatorResult:
    """定位结果。"""

    element: Any = None
    error: Optional[str] = None
    used_selector: Optional[str] = None
    source: str = "page"
    tried_selectors: list[str] = field(default_factory=list)
    matched_count: int = 0

    @property
    def found(self) -> bool:
        return self.element is not None

    def __bool__(self) -> bool:
        return self.found


def _is_explicit_drission_selector(selector: str) -> bool:
    """判断是否已经是 DrissionPage 明确定位语法。"""
    lower = selector.lower()
    prefixes = (
        "#",
        ".",
        "@",
        "tag:",
        "tag=",
        "t:",
        "t=",
        "text:",
        "text=",
        "tx:",
        "tx=",
        "xpath:",
        "xpath=",
        "x:",
        "x=",
        "css:",
        "css=",
        "c:",
        "c=",
    )
    if selector.startswith(prefixes):
        return True

    if "@@" in selector or "@|" in selector or "@!" in selector:
        return True

    if lower.startswith(("http://", "https://")):
        return False

    return False


def _looks_like_xpath(selector: str) -> bool:
    """判断是否像 XPath。"""
    xpath_signals = (
        "//",
        ".//",
        "/html",
        "(/",
        "(//",
        "./*",
    )
    return selector.startswith(xpath_signals)


def _looks_like_css(selector: str) -> bool:
    """判断是否像 CSS 选择器。"""
    if selector.startswith(("*", "[", ">", "+", "~")):
        return True

    if any(token in selector for token in ("::", ":nth-", "[", "]", " > ", " + ", " ~ ")):
        return True

    if _CSS_SIGNAL_RE.search(selector):
        return True

    if " " in selector and not selector.startswith("text:") and not selector.startswith("text="):
        parts = [p for p in selector.split(" ") if p.strip()]
        if len(parts) >= 2 and all(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", p) for p in parts):
            return True

    return False


def _normalize_single_selector(selector: str) -> str:
    """标准化单个定位符。"""
    raw = selector.strip()
    if not raw:
        return ""

    lower = raw.lower()

    if lower.startswith("xpath="):
        return f"xpath:{raw[6:]}"
    if lower.startswith("x="):
        return f"x:{raw[2:]}"
    if lower.startswith("css="):
        return f"css:{raw[4:]}"
    if lower.startswith("c="):
        return f"c:{raw[2:]}"

    if _looks_like_xpath(raw):
        return f"xpath:{raw}"

    if _is_explicit_drission_selector(raw):
        return raw

    if lower.startswith("id="):
        return f"#{raw[3:]}"
    if lower.startswith("class="):
        return f".{raw[6:]}"
    if lower.startswith("name="):
        return f"@name={raw[5:]}"

    if _looks_like_css(raw):
        return f"css:{raw}"

    return raw


def _split_selector_candidates(selector: Any) -> list[str]:
    """将 selector 拆成多个候选定位符。"""
    if selector is None:
        return []

    if isinstance(selector, (list, tuple)):
        return [str(s).strip() for s in selector if str(s).strip()]

    text = str(selector).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(s).strip() for s in data if str(s).strip()]
        except Exception:
            pass

    if "||" in text:
        return [s.strip() for s in text.split("||") if s.strip()]

    if "\n" in text:
        return [s.strip() for s in text.splitlines() if s.strip()]

    return [text]


def normalize_selectors(selector: Any) -> list[str]:
    """标准化并去重定位符列表。"""
    candidates = _split_selector_candidates(selector)
    normalized: list[str] = []
    for item in candidates:
        value = _normalize_single_selector(item)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _safe_get_elements(target: Any, selector: str, timeout: Optional[float]) -> list[Any]:
    """安全调用 eles()，避免单次失败中断整个定位流程。"""
    try:
        if timeout is None:
            result = target.eles(selector)
        else:
            result = target.eles(selector, timeout=timeout)
        return list(result) if result else []
    except Exception:
        return []


def _safe_state(ele: Any, state_name: str, default: Any = False) -> Any:
    """安全读取元素状态属性。"""
    try:
        states = getattr(ele, "states", None)
        if states is None:
            return default
        return getattr(states, state_name)
    except Exception:
        return default


def _safe_tag(ele: Any) -> str:
    """安全读取元素标签名。"""
    try:
        return str(getattr(ele, "tag", "") or "").lower()
    except Exception:
        return ""


def _safe_attr(ele: Any, name: str) -> str:
    """安全读取元素属性值。"""
    try:
        value = ele.attr(name)
        return str(value) if value is not None else ""
    except Exception:
        return ""


def _score_element(ele: Any, prefer: str) -> int:
    """按场景给元素打分。"""
    score = 0

    is_alive = bool(_safe_state(ele, "is_alive", True))
    is_displayed = bool(_safe_state(ele, "is_displayed", False))
    is_enabled = bool(_safe_state(ele, "is_enabled", True))
    is_clickable = bool(_safe_state(ele, "is_clickable", False))
    has_rect = bool(_safe_state(ele, "has_rect", False))
    is_in_viewport = bool(_safe_state(ele, "is_in_viewport", False))
    is_covered = _safe_state(ele, "is_covered", False)

    if is_alive:
        score += 2
    if has_rect:
        score += 2
    if is_displayed:
        score += 3
    if is_enabled:
        score += 3
    if is_in_viewport:
        score += 1
    if is_covered is False:
        score += 1

    tag = _safe_tag(ele)

    if prefer == "click":
        if is_clickable:
            score += 8
        if tag in {"a", "button", "input", "label", "summary"}:
            score += 2
    elif prefer == "input":
        if tag in {"input", "textarea", "select"}:
            score += 6
        if _safe_attr(ele, "contenteditable").lower() in {"", "true"}:
            score += 3
    elif prefer == "visible":
        if is_displayed:
            score += 3

    return score


def _pick_by_index(elements: list[Any], index: int) -> Any:
    """按 1 基索引选择元素，支持负数。"""
    if not elements:
        return None

    if index == 0:
        return None

    pos = index - 1 if index > 0 else index
    if -len(elements) <= pos < len(elements):
        return elements[pos]
    return None


def _pick_best_element(
    elements: list[Any],
    *,
    prefer: str,
    index: int,
    strict: bool,
) -> Any:
    """从候选元素中选出最合适的一个。"""
    if not elements:
        return None

    if index != 1:
        return _pick_by_index(elements, index)

    if strict:
        return elements[0]

    ranked = sorted(
        enumerate(elements),
        key=lambda item: (-_score_element(item[1], prefer), item[0]),
    )
    return ranked[0][1] if ranked else None


def _iter_frames(page: Any, timeout: Optional[float], max_frame_scan: int) -> Iterable[tuple[int, Any]]:
    """按需遍历页面 frame。"""
    if not hasattr(page, "get_frames"):
        return []

    frame_timeout = 0.8 if timeout is None else min(max(timeout, 0.1), 0.8)
    try:
        frames = page.get_frames(timeout=frame_timeout)
    except Exception:
        return []

    if not frames:
        return []

    return list(enumerate(list(frames)[:max_frame_scan], start=1))


def resolve_element(
    page: Any,
    selector: Any,
    timeout: float | None = None,
    *,
    index: int = 1,
    prefer: str = "auto",
    strict: bool = False,
    search_frames: bool = True,
    max_frame_scan: int = 5,
) -> LocatorResult:
    """定位单个元素。

    参数：
    - `selector`: 支持单个定位符、`||` 分隔的多候选、换行多候选、JSON 数组字符串。
    - `prefer`: `auto` / `click` / `input` / `visible`。
    - `strict`: 为 True 时不做优选评分，直接按结果顺序取元素。
    """
    candidates = normalize_selectors(selector)
    if not candidates:
        return LocatorResult(error="selector 为空")

    tried: list[str] = []

    for loc in candidates:
        tried.append(loc)
        elements = _safe_get_elements(page, loc, timeout)
        chosen = _pick_best_element(elements, prefer=prefer, index=index, strict=strict)
        if chosen is not None:
            return LocatorResult(
                element=chosen,
                used_selector=loc,
                source="page",
                tried_selectors=tried,
                matched_count=len(elements),
            )

    if search_frames:
        for frame_index, frame in _iter_frames(page, timeout, max_frame_scan):
            for loc in candidates:
                tried.append(f"frame[{frame_index}]::{loc}")
                elements = _safe_get_elements(frame, loc, timeout=0)
                chosen = _pick_best_element(elements, prefer=prefer, index=index, strict=strict)
                if chosen is not None:
                    return LocatorResult(
                        element=chosen,
                        used_selector=loc,
                        source=f"frame[{frame_index}]",
                        tried_selectors=tried,
                        matched_count=len(elements),
                    )

    return LocatorResult(
        error=f"未找到元素: {selector}",
        tried_selectors=tried,
    )


def resolve_elements(
    page: Any,
    selector: Any,
    *,
    limit: int = 50,
    timeout: float | None = None,
    search_frames: bool = False,
    max_frame_scan: int = 3,
) -> list[Any]:
    """定位多个元素，返回首个成功候选的结果集。"""
    candidates = normalize_selectors(selector)
    if not candidates:
        return []

    for loc in candidates:
        elements = _safe_get_elements(page, loc, timeout)
        if elements:
            return elements[:limit]

    if search_frames:
        for _, frame in _iter_frames(page, timeout, max_frame_scan):
            for loc in candidates:
                elements = _safe_get_elements(frame, loc, timeout=0)
                if elements:
                    return elements[:limit]

    return []


def build_selector(
    *,
    id: Optional[str] = None,
    class_name: Optional[str] = None,
    tag: Optional[str] = None,
    text: Optional[str] = None,
    text_contains: Optional[str] = None,
    attr_name: Optional[str] = None,
    attr_value: Optional[str] = None,
    attr_contains: bool = False,
) -> str:
    """按结构化参数拼装定位符。"""
    if id:
        return f"#{id}"
    if class_name:
        return f".{class_name}"
    if tag and attr_name and attr_value:
        op = ":" if attr_contains else "="
        return f"tag:{tag}@{attr_name}{op}{attr_value}"
    if attr_name and attr_value:
        op = ":" if attr_contains else "="
        return f"@{attr_name}{op}{attr_value}"
    if text:
        return f"text={text}"
    if text_contains:
        return f"text:{text_contains}"
    if tag:
        return f"tag:{tag}"
    return ""
