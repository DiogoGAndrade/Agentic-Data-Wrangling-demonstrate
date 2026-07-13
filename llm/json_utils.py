import json
import re
from typing import Any, Dict


def _try_parse(text: str) -> Dict[str, Any]:
    """Attempt json.loads; raises ValueError on failure."""
    return json.loads(text)


def _repair_truncated(text: str) -> str:
    """
    Attempt to close unclosed braces/brackets in truncated JSON.
    Common when num_predict cuts off mid-generation.

    Uses a stack-based approach to determine what needs closing.
    """
    stack = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()

    # If we ended inside a string, close it
    if in_string:
        text += '"'

    # Remove trailing comma or colon (common truncation points)
    text = re.sub(r'[,:]\s*$', '', text.rstrip())

    # Close all unclosed structures in reverse order
    text += ''.join(reversed(stack))

    return text


def _repair_common_errors(text: str) -> str:
    """
    Fix common JSON syntax errors produced by LLMs:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    - Comments (// or /* */)
    """
    # Remove single-line comments
    text = re.sub(r'//[^\n]*', '', text)
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Remove trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Try to fix single-quoted strings -> double-quoted
    # Only if there are no double quotes at all (avoids breaking apostrophes)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')

    return text


def _extract_json_block(text: str) -> str:
    """
    Extract JSON from markdown code blocks or surrounding prose.
    Handles ```json ... ``` and ``` ... ``` patterns.
    """
    patterns = [
        r'```json\s*\n?(.*?)```',
        r'```\s*\n?(.*?)```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction with multi-stage repair.

    Strategy (ordered by reliability):
    1. Direct parse of full text
    2. Extract from markdown code blocks
    3. Find first {...} block and parse
    4. Repair common syntax errors and retry
    5. Repair truncated JSON (close unclosed braces)

    Raises ValueError only if ALL strategies fail.
    """
    text = text.strip()

    # === Strategy 1: Direct parse ===
    try:
        return _try_parse(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # === Strategy 2: Extract from code blocks ===
    extracted = _extract_json_block(text)
    if extracted != text:
        try:
            return _try_parse(extracted)
        except (json.JSONDecodeError, ValueError):
            text = extracted

    # === Strategy 3: Find first {...} block ===
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start: end + 1]
        try:
            return _try_parse(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

        # === Strategy 4: Repair common errors on the candidate ===
        repaired = _repair_common_errors(candidate)
        try:
            return _try_parse(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

        # === Strategy 5: Repair truncation on the repaired candidate ===
        truncation_repaired = _repair_truncated(repaired)
        try:
            return _try_parse(truncation_repaired)
        except (json.JSONDecodeError, ValueError):
            pass

    # === Strategy 6: From first { to end (no closing brace at all) ===
    if start != -1:
        from_start = text[start:]
        repaired_full = _repair_common_errors(from_start)
        repaired_full = _repair_truncated(repaired_full)
        try:
            return _try_parse(repaired_full)
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(
        f"No valid JSON found after all repair strategies. "
        f"Text starts with: {text[:200]}..."
    )
