"""
Simple network fetch route to retrieve page content for AI context.
"""
import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

def make_router():
    router = APIRouter()

    @router.get("/net/fetch")
    async def fetch(url: str = Query(..., description="Page URL to fetch"), timeout: int = 20):
        try:
            # Basic fetch; you can add sanitization/allowlists here
            r = await httpx.AsyncClient(timeout=timeout).get(url, follow_redirects=True)
            if r.status_code != 200:
                return JSONResponse(status_code=r.status_code, content={"error": f"HTTP {r.status_code}"})
            # Return text; for large pages, consider truncation
            text = r.text
            return {"url": url, "content": text[:200_000]}  # cap content size
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return router