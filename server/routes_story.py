"""
Ops Story route: generates a human-friendly, informal summary of current status
without exposing raw logs. Uses the AI runtime (Ollama GPU) to write the story.
"""
import time
from typing import Any
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

def make_router(runtime: Any, registry: Any):
    router = APIRouter()

    @router.get("/ops/story")
    async def story(tokens: int = Query(256, ge=64, le=512)):
        try:
            # Load config for parameters (environment, risk, execution, universe)
            from pathlib import Path
            import yaml
            cfg = {}
            p = Path("config/config.yaml")
            if p.exists():
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

            env = (cfg.get("environment", {}) or {}).get("mode", "paper")
            risk = cfg.get("risk", {}) or {}
            exec_ = cfg.get("execution", {}) or {}
            uni = cfg.get("universe", {}) or {}
            features = (cfg.get("data", {}) or {}).get("features", {}) or {}

            # Precompute values (no backslashes inside f-string expressions)
            max_dd = risk.get("max_drawdown", "2%")
            per_trade = risk.get("per_trade", "0.4%")
            kill_switch = risk.get("kill_switch", True)
            exposure_limits = risk.get("exposure_limits", "default")

            order_type = exec_.get("order_type", "limit")
            bracket_orders = exec_.get("bracket_orders", True)
            toxic_time_filter = exec_.get("toxic_time_filter", True)
            rate_limit = exec_.get("rate_limit", "200/min")

            core_universe = uni.get("core", 200)
            extended_universe = uni.get("extended", 1000)

            feature_store_path = features.get("path", "data/features")

            prompt = (
                "Write a short, human-friendly status update for a trading operator.\n"
                "Plain English, upbeat, no jargon. Use bullets when listing parameters.\n"
                "Include: environment, risk caps, execution setup, universe size, and what’s next.\n\n"
                f"Environment: {env}\n"
                f"Risk: max_dd={max_dd}, per_trade={per_trade}, "
                f"kill_switch={kill_switch}, exposure_limits={exposure_limits}\n"
                f"Execution: order_type={order_type}, bracket_orders={bracket_orders}, "
                f"toxic_time_filter={toxic_time_filter}, rate_limit={rate_limit}\n"
                f"Universe: core={core_universe}, extended={extended_universe}\n"
                f"Feature store: {feature_store_path}\n\n"
                "Close with a one-liner about today’s focus and readiness."
            )

            # Use the default alias explicitly
            alias = getattr(registry, "get_default_alias", lambda: None)()
            text = runtime.generate(alias=alias, prompt=prompt, options={"num_predict": int(tokens)}, stream=False)
            return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "story": text}
        except Exception as e:
            # Include the alias to make debugging easier
            alias = getattr(registry, "get_default_alias", lambda: None)()
            return JSONResponse(status_code=500, content={"error": f"{e}", "alias": alias})

    return router