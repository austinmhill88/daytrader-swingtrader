"""
Parameters route: reads config/config.yaml to present key trading settings
in a simple structure for the UI.
"""
from pathlib import Path
from typing import Any, Dict
import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse

def _safe_get(d: Dict[str, Any], path: str, default=None):
  cur = d
  for key in path.split('.'):
    if not isinstance(cur, dict) or key not in cur:
      return default
    cur = cur[key]
  return cur

def make_router(cfg_path: str = "config/config.yaml"):
  router = APIRouter()

  @router.get("/ui/params")
  async def params():
    try:
      p = Path(cfg_path)
      data: Dict[str, Any] = {}
      if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
      # Build response
      resp = {
        "env": _safe_get(data, "environment.mode", "paper"),
        "risk": {
          "max_dd": _safe_get(data, "risk.max_drawdown", "2%"),
          "per_trade": _safe_get(data, "risk.per_trade", "0.4%"),
          "kill_switch": bool(_safe_get(data, "risk.kill_switch", True)),
          "exposure": _safe_get(data, "risk.exposure_limits", "default"),
        },
        "exec": {
          "order_type": _safe_get(data, "execution.order_type", "limit"),
          "bracket_orders": bool(_safe_get(data, "execution.bracket_orders", True)),
          "toxic_time": bool(_safe_get(data, "execution.toxic_time_filter", True)),
          "rate_limit": _safe_get(data, "execution.rate_limit", "200/min"),
        },
        "data": {
          "core": _safe_get(data, "universe.core", 200),
          "extended": _safe_get(data, "universe.extended", 1000),
          "historical": _safe_get(data, "data.historical.enabled", True),
          "feature_store": _safe_get(data, "data.features.path", "data\\features"),
        },
      }
      return resp
    except Exception as e:
      return JSONResponse(status_code=500, content={"error": str(e)})
  return router