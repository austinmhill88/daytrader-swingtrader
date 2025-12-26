"""
Integrated app entrypoint (Ollama GPU backend) with trading-focused dashboard + chat memory routes.
"""
import os
from pathlib import Path
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from loguru import logger

# Load .env (optional, for SEARCH_PROVIDER and keys)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Backends
from .model_runtime import ModelRegistry as LocalRegistry, AIModelRuntime as LocalRuntime
from .model_runtime_ollama import OllamaRegistry, OllamaRuntime

# Routers
from .routes_model import make_router as make_model_router
from .routes_web import make_router as make_web_router
from .routes_fs import make_router as make_fs_router
from .routes_ops import make_router as make_ops_router
from .routes_aiops import make_router as make_aiops_router
from .routes_diag import make_router as make_diag_router
from .routes_params import make_router as make_params_router
from .routes_dashboard import make_router as make_dashboard_router
from .routes_chat import make_router as make_chat_router
from .chat_memory import ChatMemory
from .trading_controller import TradingController

def load_ai_config():
    cfg_path = Path("config/ai-coder.yaml")
    if not cfg_path.exists():
        raise RuntimeError(f"Missing {cfg_path}. Create it from the template.")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def load_persona() -> str:
    p = Path("config/ai-persona.md")
    return p.read_text(encoding="utf-8") if p.exists() else ""

def create_app() -> FastAPI:
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/integrated.log", rotation="10 MB", retention="30 days", level="INFO")

    # Log web search provider status (no secrets printed)
    provider = (os.getenv("SEARCH_PROVIDER") or "").lower() or "none"
    g_key_set = bool(os.getenv("GOOGLE_API_KEY"))
    g_cx_set = bool(os.getenv("GOOGLE_CSE_ID"))
    b_key_set = bool(os.getenv("BING_API_KEY"))
    br_key_set = bool(os.getenv("BRAVE_API_KEY"))
    serp_key_set = bool(os.getenv("SERPAPI_KEY"))
    logger.info(f"Web search provider: {provider} | Google key: {g_key_set} | Google cx: {g_cx_set} | Bing key: {b_key_set} | Brave key: {br_key_set} | SerpAPI key: {serp_key_set}")

    ai_cfg = load_ai_config()
    backend = (ai_cfg.get("backend") or "ollama").lower()
    models_cfg = ai_cfg.get("models", []) or []
    default_alias = ai_cfg.get("default_model")
    runtime_defaults = ai_cfg.get("runtime_defaults", {}) or {}
    persona = load_persona()

    if backend == "ollama":
        base_url = ai_cfg.get("ollama_base_url") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
        registry = OllamaRegistry(base_url=base_url)
        registry.load_specs(models_cfg, default_alias)
        runtime = OllamaRuntime(registry, runtime_defaults)
        logger.info(f"Backend: Ollama @ {base_url}")
    else:
        registry = LocalRegistry()
        registry.load_specs(models_cfg, default_alias)
        runtime = LocalRuntime(registry, runtime_defaults)
        logger.info("Backend: local llama.cpp")

    trading_cfg_path = os.environ.get("TRADING_CONFIG_PATH", "config/config.yaml")
    controller = TradingController(trading_cfg_path)

    allow_network = bool(ai_cfg.get("allow_network", True))
    allow_html = bool(ai_cfg.get("allow_html", False))
    root_dir = str(ai_cfg.get("root_dir") or Path(".").resolve())
    allow_write = bool(ai_cfg.get("allow_write", False))
    allowed_globs = list(ai_cfg.get("allowed_globs", []) or [])
    denied_globs = list(ai_cfg.get("denied_globs", []) or [])

    # Prometheus branding
    app = FastAPI(title="Prometheus", version="1.0.1")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # API routes (AI + trading)
    app.include_router(make_model_router(registry, runtime, default_alias, system_prompt=persona))
    app.include_router(make_aiops_router(registry, runtime, default_alias))
    app.include_router(make_diag_router(registry))
    app.include_router(make_web_router(allow_network, allow_html))
    app.include_router(make_fs_router(root_dir, allow_write, allowed_globs, denied_globs))
    app.include_router(make_ops_router(controller, logs_dir="logs"))

    # Trading UI helper routes
    app.include_router(make_params_router(cfg_path=trading_cfg_path))
    app.include_router(make_dashboard_router(controller))

    # Chat memory routes (stateful assistant with facts + market grounding + optional web search)
    memory = ChatMemory(max_messages=120)
    app.include_router(make_chat_router(runtime, registry, memory, logs_path="logs/integrated.log", controller=controller))

    # Redirect "/" to the UI
    @app.get("/")
    async def root():
        return RedirectResponse("/ui")

    web_dir = Path("web")
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")
        logger.info(f"Static UI mounted at /ui (path={web_dir.resolve()})")

    logger.info(f"Integrated app ready | Backend={backend} | Default model: {default_alias} | Trading config: {trading_cfg_path}")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Use the same port you’ve been using
    uvicorn.run("server.app_integrated:app", host="127.0.0.1", port=8001, reload=False)