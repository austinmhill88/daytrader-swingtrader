"""
Model API routes - Ollama-compatible endpoints for chat and generation.
"""
import json
import time
from typing import Dict, Any, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .model_runtime import AIModelRuntime, ModelRegistry


def make_router(registry: ModelRegistry, runtime: AIModelRuntime, default_alias: Optional[str]):
    """
    Create model API router.
    
    Args:
        registry: Model registry
        runtime: AI model runtime
        default_alias: Default model alias
        
    Returns:
        FastAPI router with model endpoints
    """
    router = APIRouter()

    @router.get("/api/version")
    async def version():
        """Get server version and backend info."""
        return {
            "version": "local-0.1.0", 
            "backend": "llama.cpp", 
            "default_model": default_alias
        }

    @router.get("/api/tags")
    async def tags():
        """List available models (Ollama-compatible)."""
        models = [
            {"name": alias, "modified_at": "", "size": 0} 
            for alias in registry.list_aliases()
        ]
        return {"models": models}

    @router.post("/api/models/load")
    async def models_load(req: Dict[str, Any]):
        """Explicitly load a model by alias."""
        alias = (req or {}).get("alias")
        try:
            _ = registry.ensure_loaded(alias)
            return {"ok": True, "alias": alias}
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    @router.post("/api/chat")
    async def chat(req: Dict[str, Any]):
        """
        Chat completion endpoint (Ollama-compatible).
        
        Request body:
            - model: Model alias (optional, uses default)
            - messages: List of {role, content} messages
            - options: Runtime options (temperature, etc.)
            - stream: Boolean for streaming response
        """
        alias = req.get("model") or default_alias
        messages = req.get("messages") or []
        options = req.get("options") or {}
        stream = bool(req.get("stream", False))

        # Normalize roles/content
        norm = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = m.get("content") or ""
            norm.append({"role": role, "content": content})

        if stream:
            def ndjson():
                # Initial line with metadata
                yield json.dumps({
                    "model": alias, 
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "message": {"role": "assistant", "content": ""}
                }) + "\n"
                
                try:
                    for delta in runtime.chat(alias, norm, options, stream=True):
                        yield json.dumps({
                            "message": {"role": "assistant", "content": delta}
                        }) + "\n"
                    yield json.dumps({"done": True}) + "\n"
                except Exception as e:
                    yield json.dumps({"error": str(e)}) + "\n"
            
            return StreamingResponse(ndjson(), media_type="application/x-ndjson")

        # Non-streaming response
        try:
            text = runtime.chat(alias, norm, options, stream=False)
            return {
                "model": alias, 
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": {"role": "assistant", "content": text}
            }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.post("/api/generate")
    async def generate(req: Dict[str, Any]):
        """
        Text generation endpoint (Ollama-compatible).
        
        Request body:
            - model: Model alias (optional, uses default)
            - prompt: Prompt text
            - options: Runtime options (temperature, etc.)
            - stream: Boolean for streaming response
        """
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

        # Non-streaming response
        try:
            text = runtime.generate(alias, prompt, options, stream=False)
            return {"response": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router
