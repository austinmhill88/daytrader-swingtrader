"""
Model API routes - Ollama-compatible endpoints for chat and generation.
Forces trader persona and auto-injects recent logs into chat context.
"""
import json
import time
import os
import io
from typing import Dict, Any, Optional, List

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .model_runtime import AIModelRuntime, ModelRegistry

# Keep prompts small to avoid context overflow
MAX_CHARS_PER_LOG = 3000
DEFAULT_LOG_LINES = 60

def _read_tail(path: str, lines: int = DEFAULT_LOG_LINES) -> str:
    try:
        if not os.path.exists(path):
            return ""
        with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()[-lines:]
            text = "".join(data)
            return text[-MAX_CHARS_PER_LOG:]
    except Exception:
        return ""

def make_router(
    registry: ModelRegistry,
    runtime: AIModelRuntime,
    default_alias: Optional[str],
    system_prompt: Optional[str] = None
):
    """
    Create model API router.

    Args:
        registry: Model registry
        runtime: AI model runtime
        default_alias: Default model alias
        system_prompt: Trader persona (Markdown string) injected in every chat
    """
    router = APIRouter()

    def _compose_messages(user_messages: List[Dict[str, str]], log_lines: int) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        # Always inject trader persona
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        # Add a strict behavior directive (avoid generic disclaimers)
        msgs.append({
            "role": "system",
            "content": (
                "Operate as an on-desk trading assistant. "
                "Do not use generic disclaimers. "
                "Use logs as system context. Be concise and actionable."
            )
        })

        # Auto-attach recent logs
        runtime_log = _read_tail("logs/runtime.log", log_lines)
        errors_log = _read_tail("logs/errors.log", log_lines)
        trades_log = _read_tail("logs/trades.log", log_lines)
        context_parts = []
        if runtime_log: context_parts.append(f"=== runtime.log (last {log_lines}) ===\n{runtime_log}")
        if errors_log: context_parts.append(f"=== errors.log (last {log_lines}) ===\n{errors_log}")
        if trades_log: context_parts.append(f"=== trades.log (last {log_lines}) ===\n{trades_log}")
        if context_parts:
            msgs.append({
                "role": "system",
                "content": "Operational context from logs:\n" + "\n\n".join(context_parts)
            })

        # Normalize and append user messages
        for m in user_messages or []:
            role = (m.get("role") or "user").lower()
            content = m.get("content") or ""
            msgs.append({"role": role, "content": content})
        return msgs

    @router.get("/api/version")
    async def version():
        return {"version": "local-0.4.1", "backend": "llama.cpp", "default_model": default_alias}

    @router.get("/api/tags")
    async def tags():
        models = [{"name": alias, "modified_at": "", "size": 0} for alias in registry.list_aliases()]
        return {"models": models}

    @router.post("/api/models/load")
    async def models_load(req: Dict[str, Any]):
        alias = (req or {}).get("alias") or default_alias
        try:
            _ = registry.ensure_loaded(alias)
            return {"ok": True, "alias": alias}
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @router.post("/api/chat")
    async def chat(req: Dict[str, Any]):
        """
        Chat completion endpoint (Ollama-compatible).
        Always injects persona and recent logs.
        Request body:
            - model: Model alias (optional, uses default)
            - messages: List of {role, content}
            - options: Runtime options
            - stream: Boolean to stream
            - log_lines: Int (default 60) number of lines to include from each log
        """
        alias = req.get("model") or default_alias
        messages = req.get("messages") or []
        options = req.get("options") or {}
        stream = bool(req.get("stream", False))
        log_lines = int(req.get("log_lines") or DEFAULT_LOG_LINES)

        final_messages = _compose_messages(messages, log_lines)

        if stream:
            def ndjson():
                yield json.dumps({
                    "model": alias,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "message": {"role": "assistant", "content": ""}
                }) + "\n"
                try:
                    for delta in runtime.chat(alias, final_messages, options, stream=True):
                        yield json.dumps({"message": {"role": "assistant", "content": delta}}) + "\n"
                    yield json.dumps({"done": True}) + "\n"
                except Exception as e:
                    yield json.dumps({"error": str(e)}) + "\n"
            return StreamingResponse(ndjson(), media_type="application/x-ndjson")

        try:
            text = runtime.chat(alias, final_messages, options, stream=False)
            return {
                "model": alias,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": {"role": "assistant", "content": text}
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.post("/api/generate")
    async def generate(req: Dict[str, Any]):
        alias = req.get("model") or default_alias
        prompt = req.get("prompt") or ""
        options = req.get("options") or {}
        stream = bool(req.get("stream", False))

        if stream:
            def ndjson():
                try:
                    for delta in runtime.generate(alias, prompt, options, stream=True):
                        yield json.dumps({"response": delta}) + "\n"
                    yield json.dumps({"done": True}) + "\n"
                except Exception as e:
                    yield json.dumps({"error": str(e)}) + "\n"
            return StreamingResponse(ndjson(), media_type="application/x-ndjson")

        try:
            text = runtime.generate(alias, prompt, options, stream=False)
            return {"response": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router