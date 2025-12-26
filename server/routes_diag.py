"""
Diagnostic route so you can see what model backend is configured at runtime.
Works for both local llama.cpp and Ollama backends.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

def make_router(registry):
    router = APIRouter()

    @router.get("/ai/diag")
    async def diag():
        try:
            default = getattr(registry, "get_default_alias", lambda: None)()
            aliases = getattr(registry, "list_aliases", lambda: [])()
            specs = []
            # Try to read specs from registry._specs (used by both local and ollama registries)
            spec_map = getattr(registry, "_specs", {}) or {}
            for alias in aliases:
                spec = spec_map.get(alias)
                if spec is None:
                    specs.append({"alias": alias})
                    continue
                # Build a simple dict of public attributes (strings/ints/bools)
                out = {"alias": alias}
                for k in dir(spec):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(spec, k)
                    except Exception:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        out[k] = v
                specs.append(out)
            return {"default_model": default, "models": specs}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router