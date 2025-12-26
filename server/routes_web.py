"""
Web access routes - Fetch pages and optional search via SearxNG.
"""
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse


def make_router(allow_network: bool, allow_html: bool = True):
    """
    Create web access router.
    
    Args:
        allow_network: Whether network access is allowed
        allow_html: If False, attempt to return text-only (basic stripping)
        
    Returns:
        FastAPI router with web endpoints
    """
    router = APIRouter()

    @router.get("/web/fetch")
    async def fetch(url: str = Query(..., description="URL to fetch")):
        """
        Fetch a web page without restrictions.
        
        Args:
            url: URL to fetch
            
        Returns:
            Page text, status, headers, and truncation info
        """
        if not allow_network:
            return JSONResponse(
                status_code=403, 
                content={"error": "Network disabled by config"}
            )
        
        # Simple robots.txt check (best-effort, non-blocking)
        try:
            parsed = httpx.URL(url)
            robots_url = f"{parsed.scheme}://{parsed.host}/robots.txt"
            async with httpx.AsyncClient(timeout=10) as client:
                robots = await client.get(robots_url)
                if robots.status_code == 200 and "Disallow:" in robots.text:
                    # Log only; do not block (user explicitly requested fetch)
                    pass
        except Exception:
            pass  # Ignore robots.txt errors

        # Fetch the actual page
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "LocalAI/0.1"})
                text = r.text
                
                # Trim excessive content
                truncated = False
                if len(text) > 1_500_000:
                    text = text[:1_500_000]
                    truncated = True

                if not allow_html:
                    # Naive HTML strip: remove tags; keep text content
                    import re
                    text = re.sub(r"<script.*?</script>", "", text, flags=re.S | re.I)
                    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
                    text = re.sub(r"<[^>]+>", "", text)

                return {
                    "url": url,
                    "status": r.status_code,
                    "headers": dict(r.headers),
                    "truncated": truncated,
                    "text": text,
                }
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/web/search")
    async def search(
        q: str = Query(..., description="Search query"), 
        source: str = Query("searxng", description="Search source")
    ):
        """
        Search the web via SearxNG (if configured).
        
        Args:
            q: Search query
            source: Search source (currently only "searxng")
            
        Returns:
            Search results with title, url, snippet
        """
        if not allow_network:
            return JSONResponse(
                status_code=403, 
                content={"error": "Network disabled by config"}
            )
        
        searxng_url = os.environ.get("SEARXNG_URL")
        
        if source == "searxng" and searxng_url:
            try:
                async with httpx.AsyncClient(timeout=12) as client:
                    r = await client.get(
                        f"{searxng_url}/search", 
                        params={"q": q, "format": "json", "categories": "general"}
                    )
                    
                    if r.status_code != 200:
                        return JSONResponse(
                            status_code=r.status_code, 
                            content={"error": r.text}
                        )
                    
                    data = r.json() or {}
                    results = []
                    
                    for item in data.get("results", [])[:10]:
                        results.append({
                            "title": item.get("title") or "",
                            "url": item.get("url") or "",
                            "snippet": item.get("content") or "",
                        })
                    
                    return {"query": q, "results": results}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
        
        return JSONResponse(
            status_code=400, 
            content={
                "error": "Search not configured (set SEARXNG_URL) or unsupported source"
            }
        )

    return router