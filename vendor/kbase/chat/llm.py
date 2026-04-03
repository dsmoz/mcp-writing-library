"""LLM streaming for chat endpoints.

Provider routing:
  1. Anthropic  — if ANTHROPIC_API_KEY set and model starts with "claude-" (or no model override)
                  Uses anthropic SDK with prompt caching on the system block.
  2. OpenRouter — if OPENROUTER_API_KEY set. Raw HTTP POST with stream=True.
  3. LM Studio  — fallback. LLM_API_URL + LLM_MODEL env vars required.
  4. Error      — yields {"error": "..."} if nothing is configured.
"""

import json
import os
from typing import Iterator


def _build_system_prompt(context: str) -> str:
    return (
        "You are a research assistant. Answer based only on the provided context.\n"
        "When citing passages, use [1], [2] etc. to reference them.\n\n"
        f"Context:\n{context}"
    )


def _is_anthropic_model(model: str | None) -> bool:
    if not model:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return model.startswith("claude-")


def stream_llm_response(
    question: str,
    context: str,
    history: list[dict],
    sources: list[dict],
) -> Iterator[dict]:
    """Stream LLM response tokens.

    Yields:
        {"token": "..."}  — one per token
        {"done": True, "sources": [...]}  — on completion
        {"error": "..."}  — on failure
    """
    model = os.getenv("CHAT_MODEL", "")  # empty = auto-detect from API keys
    system_text = _build_system_prompt(context)

    # Build OpenAI-style message list (used by OpenRouter + LM Studio paths)
    oa_messages = [{"role": "system", "content": system_text}]
    oa_messages += [{"role": m["role"], "content": m["content"]} for m in history]
    oa_messages.append({"role": "user", "content": question})

    # ── 1. Anthropic ──────────────────────────────────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and _is_anthropic_model(model or None):
        try:
            import anthropic as _anthropic
            anthropic_model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            client = _anthropic.Anthropic(api_key=anthropic_key)
            chat_messages = [m for m in oa_messages if m["role"] != "system"]
            stream_ctx = client.messages.stream(
                model=anthropic_model,
                max_tokens=2048,
                system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
                messages=chat_messages,
            )
        except Exception:
            pass  # setup failed, fall through to next provider
        else:
            # Streaming started — errors here must NOT fall through
            try:
                with stream_ctx as stream:
                    for text in stream.text_stream:
                        yield {"token": text}
                yield {"done": True, "sources": sources}
            except Exception as e:
                yield {"error": f"Anthropic stream error: {e}"}
            return

    # ── 2. OpenRouter ─────────────────────────────────────────────────────────
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        import requests as _req
        openrouter_model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it")
        try:
            resp = _req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {openrouter_key}", "Content-Type": "application/json"},
                json={"model": openrouter_model, "messages": oa_messages, "stream": True, "max_tokens": 2048},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()
        except Exception:
            pass  # setup failed, fall through to LM Studio
        else:
            # Streaming started — errors here must NOT fall through
            try:
                for line in resp.iter_lines():
                    if not line or line == b"data: [DONE]":
                        continue
                    if line.startswith(b"data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0].get("delta", {}).get("content", "")
                            if token:
                                yield {"token": token}
                        except Exception:
                            pass
                yield {"done": True, "sources": sources}
            except Exception as e:
                yield {"error": f"OpenRouter stream error: {e}"}
            return

    # ── 3. LM Studio ─────────────────────────────────────────────────────────
    lm_url = os.getenv("LLM_API_URL")
    lm_model = os.getenv("LLM_MODEL")
    if lm_url and lm_model:
        import requests as _req
        try:
            resp = _req.post(
                f"{lm_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'local')}",
                         "Content-Type": "application/json"},
                json={"model": lm_model, "messages": oa_messages, "stream": True, "max_tokens": 2048},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()
        except Exception:
            pass  # setup failed, fall through
        else:
            # Streaming started — errors here must NOT fall through
            try:
                for line in resp.iter_lines():
                    if not line or line == b"data: [DONE]":
                        continue
                    if line.startswith(b"data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk["choices"][0].get("delta", {}).get("content", "")
                            if token:
                                yield {"token": token}
                        except Exception:
                            pass
                yield {"done": True, "sources": sources}
            except Exception as e:
                yield {"error": f"LM Studio stream error: {e}"}
            return

    # ── 4. No provider configured ─────────────────────────────────────────────
    yield {"error": "No LLM configured. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or LLM_API_URL+LLM_MODEL."}
