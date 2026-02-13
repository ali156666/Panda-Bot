"""Page snapshot extraction for LLM understanding.

Provides methods to extract page content in different formats for Agent consumption.
"""
from __future__ import annotations

import re
from typing import Any, Optional


def get_text_snapshot(page, max_chars: int = 50000) -> str:
    """Extract simplified text content from page.
    
    Args:
        page: ChromiumPage instance.
        max_chars: Maximum characters to return.
    
    Returns:
        Text representation of page content.
    """
    try:
        # Get page info
        url = page.url or ""
        title = page.title or ""
        
        # Get text content
        body = page.ele("tag:body")
        if body:
            text = body.text or ""
        else:
            text = page.html or ""
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Build snapshot
        snapshot = f"URL: {url}\nTitle: {title}\n\nContent:\n{text}"
        
        if len(snapshot) > max_chars:
            snapshot = snapshot[:max_chars] + "\n...[truncated]"
        
        return snapshot
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def get_interactive_snapshot(page, max_chars: int = 50000) -> str:
    """Extract interactive elements for action planning.
    
    Returns a structured list of clickable/input elements with identifiers.
    
    Args:
        page: ChromiumPage instance.
        max_chars: Maximum characters to return.
    
    Returns:
        Formatted list of interactive elements.
    """
    try:
        url = page.url or ""
        title = page.title or ""
        
        lines = [f"URL: {url}", f"Title: {title}", "", "Interactive Elements:"]
        
        # Find interactive elements
        selectors = [
            ("Links", "tag:a"),
            ("Buttons", "tag:button"),
            ("Inputs", "tag:input"),
            ("Textareas", "tag:textarea"),
            ("Selects", "tag:select"),
        ]
        
        ref_id = 1
        for category, selector in selectors:
            try:
                elements = page.eles(selector)
                if not elements:
                    continue
                    
                lines.append(f"\n[{category}]")
                for ele in elements[:50]:  # Limit per category
                    try:
                        # Get identifier info
                        tag = ele.tag or "element"
                        text = (ele.text or "")[:100].strip()
                        ele_id = ele.attr("id") or ""
                        ele_class = ele.attr("class") or ""
                        ele_name = ele.attr("name") or ""
                        ele_type = ele.attr("type") or ""
                        href = ele.attr("href") or ""
                        
                        # Build description
                        desc_parts = []
                        if text:
                            desc_parts.append(f'text="{text}"')
                        if ele_id:
                            desc_parts.append(f'id="{ele_id}"')
                        if ele_name:
                            desc_parts.append(f'name="{ele_name}"')
                        if ele_type:
                            desc_parts.append(f'type="{ele_type}"')
                        if href and len(href) < 100:
                            desc_parts.append(f'href="{href}"')
                        if ele_class and len(ele_class) < 50:
                            desc_parts.append(f'class="{ele_class}"')
                        
                        desc = " ".join(desc_parts) if desc_parts else f"<{tag}>"
                        lines.append(f"  [ref:{ref_id}] {desc}")
                        ref_id += 1
                    except Exception:
                        continue
            except Exception:
                continue
        
        snapshot = "\n".join(lines)
        if len(snapshot) > max_chars:
            snapshot = snapshot[:max_chars] + "\n...[truncated]"
        
        return snapshot
    except Exception as e:
        return f"Error extracting interactive elements: {str(e)}"


def get_aria_snapshot(page, max_chars: int = 50000) -> str:
    """Extract accessibility tree representation.
    
    Provides ARIA-like view of page structure.
    
    Args:
        page: ChromiumPage instance.
        max_chars: Maximum characters to return.
    
    Returns:
        ARIA tree representation.
    """
    try:
        url = page.url or ""
        title = page.title or ""
        
        lines = [f"URL: {url}", f"Title: {title}", "", "ARIA Structure:"]
        
        # Build simplified ARIA tree
        def process_element(ele, depth: int = 0, max_depth: int = 5) -> list[str]:
            if depth > max_depth:
                return []
            
            result = []
            indent = "  " * depth
            
            try:
                tag = ele.tag or ""
                role = ele.attr("role") or ""
                aria_label = ele.attr("aria-label") or ""
                text = (ele.text or "")[:80].strip()
                
                # Determine role
                if role:
                    node_role = role
                elif tag in ("a",):
                    node_role = "link"
                elif tag in ("button",):
                    node_role = "button"
                elif tag in ("input",):
                    input_type = ele.attr("type") or "text"
                    node_role = f"textbox" if input_type in ("text", "email", "password") else input_type
                elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    node_role = f"heading"
                elif tag in ("img",):
                    node_role = "image"
                elif tag in ("nav",):
                    node_role = "navigation"
                elif tag in ("main",):
                    node_role = "main"
                elif tag in ("header",):
                    node_role = "banner"
                elif tag in ("footer",):
                    node_role = "contentinfo"
                else:
                    node_role = tag
                
                # Build label
                label = aria_label or text or ""
                if label:
                    result.append(f"{indent}{node_role}: {label}")
                else:
                    result.append(f"{indent}{node_role}")
                
                # Process children for structural elements
                if tag in ("div", "section", "article", "main", "nav", "header", "footer", "ul", "ol"):
                    try:
                        children = ele.children()
                        for child in children[:20]:
                            result.extend(process_element(child, depth + 1, max_depth))
                    except Exception:
                        pass
            except Exception:
                pass
            
            return result
        
        # Process main landmarks
        landmarks = ["tag:main", "tag:nav", "tag:header", "tag:footer", "tag:article"]
        for selector in landmarks:
            try:
                elements = page.eles(selector)
                for ele in elements[:5]:
                    lines.extend(process_element(ele, depth=1))
            except Exception:
                continue
        
        # If no landmarks, process body
        if len(lines) <= 4:
            try:
                body = page.ele("tag:body")
                if body:
                    lines.extend(process_element(body, depth=1, max_depth=3))
            except Exception:
                pass
        
        snapshot = "\n".join(lines)
        if len(snapshot) > max_chars:
            snapshot = snapshot[:max_chars] + "\n...[truncated]"
        
        return snapshot
    except Exception as e:
        return f"Error extracting ARIA tree: {str(e)}"


def get_snapshot(
    page,
    format: str = "interactive",
    max_chars: int = 50000,
) -> str:
    """Get page snapshot in specified format.
    
    Args:
        page: ChromiumPage instance.
        format: 'text', 'interactive', or 'aria'.
        max_chars: Maximum characters.
    
    Returns:
        Page snapshot string.
    """
    if format == "text":
        return get_text_snapshot(page, max_chars)
    elif format == "aria":
        return get_aria_snapshot(page, max_chars)
    else:  # default to interactive
        return get_interactive_snapshot(page, max_chars)
