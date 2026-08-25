import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

_OPENAI_CLIENT = None


def get_openai_client():
    """Return an OpenAI-compatible client.

    For OpenRouter, set OPENROUTER_API_KEY. Optional:
      OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
      OPENROUTER_SITE_URL=https://your-site.example
      OPENROUTER_APP_NAME=RULERS Demo

    For direct OpenAI API, set OPENAI_API_KEY and OPENAI_BASE_URL if needed.
    """
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY for OpenRouter or OPENAI_API_KEY for OpenAI."
        )

    base_url = (
        os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ("https://openrouter.ai/api/v1" if os.environ.get("OPENROUTER_API_KEY") else None)
    )
    headers = {}
    if os.environ.get("OPENROUTER_API_KEY"):
        headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com/anonymous/rulers-demo"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "RULERS ASAP2.0 Demo"),
        }

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if headers:
        kwargs["default_headers"] = headers
    _OPENAI_CLIENT = OpenAI(**kwargs)
    return _OPENAI_CLIENT


def _extract_first_json_object(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^```\s*", "", s).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    start = s.find("{")
    if start == -1:
        return None
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def openai_structured(
    model: str,
    system: str,
    user: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 2000,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Call an OpenAI-compatible chat endpoint with structured JSON output.

    The function first tries strict json_schema output. If the provider/model does not
    support it, the exception is retried; users can switch provider/model or prebuild
    rubric bundles. This keeps the demo close to the paper implementation.
    """
    client = get_openai_client()
    last_err = None
    n_calls = 0
    for attempt in range(1, max_retries + 1):
        try:
            n_calls += 1
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
            )
            choice = resp.choices[0]
            finish = getattr(choice, "finish_reason", "") or ""
            msg = choice.message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"Model refusal: {msg.refusal}")
            content = (msg.content or "").strip()
            try:
                obj = json.loads(content) if content else {}
            except Exception:
                j = _extract_first_json_object(content)
                if not j:
                    raise
                obj = json.loads(j)

            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "n_calls": n_calls}
            if getattr(resp, "usage", None):
                usage.update({
                    "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
                    "n_calls": n_calls,
                })
            if finish == "length" and attempt < max_retries:
                max_tokens = int(max_tokens * 1.35) + 200
                continue
            return obj, usage, finish
        except Exception as exc:
            last_err = exc
            time.sleep(1.0 * attempt)
            max_tokens = int(max_tokens * 1.25) + 100
    raise RuntimeError(f"openai_structured failed after retries: {last_err}")
