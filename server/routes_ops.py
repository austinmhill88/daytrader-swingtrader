"""
Ops routes: start/stop trading, status, and convenient log tails.
"""
from typing import Dict, Any
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import os

def make_router(controller, logs_dir: str):
    router = APIRouter()

    @router.post("/ops/start")
    async def start():
        res = controller.start()
        if not res.get("ok"):
            return JSONResponse(status_code=500, content=res)
        return res

    @router.post("/ops/stop")
    async def stop():
        res = controller.stop()
        if not res.get("ok"):
            return JSONResponse(status_code=500, content=res)
        return res

    @router.get("/ops/status")
    async def status():
        return controller.status()

    @router.get("/ops/logs")
    async def logs(name: str = Query("runtime.log", description="runtime.log | errors.log | trades.log"), lines: int = 200):
        import io
        p = os.path.join(logs_dir, name)
        if not os.path.exists(p):
            return JSONResponse(status_code=404, content={"error": f"Missing {p}"})
        try:
            with io.open(p, "r", encoding="utf-8", errors="ignore") as f:
                data = f.readlines()[-lines:]
            return {"file": name, "lines": lines, "content": "".join(data)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.post("/ops/clear_logs")
    async def clear_logs():
        cleared = []
        for fn in ["runtime.log", "errors.log", "trades.log"]:
            p = os.path.join(logs_dir, fn)
            try:
                if os.path.exists(p):
                    open(p, "w").close()
                    cleared.append(fn)
            except Exception:
                pass
        return {"ok": True, "cleared": cleared}

    return router