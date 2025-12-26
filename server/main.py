"""
Main AI Server application - FastAPI with Ollama-compatible API.
Single process; no browser rendering. Web router can block HTML by config.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .model_runtime import ModelRegistry, AIModelRuntime
from .routes_model import make_router as make_model_router
from .routes_web import make_router as make_web_router
from .routes_fs import make_router as make_fs_router


def load_config() -> Dict[str, Any]:
    cfg_path = Path("config/ai-coder.yaml")
    if not cfg_path.exists():
        raise RuntimeError(f"Missing {cfg_path}. Create it from the template.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Expand env vars in paths
    if "models" in config:
        for model in config["models"]:
            if "gguf_path" in model:
                model["gguf_path"] = os.path.expandvars(model["gguf_path"])
    if "root_dir" in config:
        config["root_dir"] = os.path.expandvars(config["root_dir"])
    return config


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        "logs/ai_tools.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def make_app() -> FastAPI:
    setup_logging()
    logger.info("Loading configuration...")
    cfg = load_config()

    # Model runtime
    models_cfg = cfg.get("models", []) or []
    default_alias = cfg.get("default_model")
    runtime_defaults = cfg.get("runtime_defaults", {}) or {}
    registry = ModelRegistry()
    registry.load_specs(models_cfg, default_alias)
    runtime = AIModelRuntime(registry, runtime_defaults)

    logger.info(f"Registered {len(registry.list_aliases())} model(s)")
    logger.info(f"Default model: {default_alias}")

    # Policy
    allow_network = bool(cfg.get("allow_network", True))
    allow_html = bool(cfg.get("allow_html", False))  # default: block HTML bodies
    root_dir = str(cfg.get("root_dir") or Path(".").resolve())
    allow_write = bool(cfg.get("allow_write", False))
    allowed_globs = list(cfg.get("allowed_globs", []) or [])
    denied_globs = list(cfg.get("denied_globs", []) or [])

    logger.info(f"Network access: {allow_network}")
    logger.info(f"Allow HTML bodies: {allow_html}")
    logger.info(f"File sandbox root: {root_dir}")
    logger.info(f"Write access: {allow_write}")

    app = FastAPI(
        title="Local AI Runtime & Tools",
        version="0.1.1",
        description="Ollama-compatible API with pure HTTP web access (no browser) and file sandbox",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(make_model_router(registry, runtime, default_alias))
    app.include_router(make_web_router(allow_network, allow_html))
    app.include_router(make_fs_router(root_dir, allow_write, allowed_globs, denied_globs))

    @app.get("/")
    async def root():
        return {
            "ok": True,
            "service": "AI Runtime & Tools",
            "version": "0.1.1",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "models_loaded": len(registry._instances),
            "models_available": len(registry.list_aliases()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    logger.info("AI Runtime server initialized successfully")
    return app


def main():
    try:
        logger.info("=" * 80)
        logger.info("Starting AI Runtime & Tools Server")
        logger.info("=" * 80)
        app = make_app()
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()