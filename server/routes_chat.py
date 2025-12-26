"""
Chat routes with session memory, persisted facts, market grounding,
automatic web research (search + page fetch + summarization with citations),
and direct live quote for current price questions.

Prevents "User:/Assistant:" role-play via stop sequences and reply cleanup.
"""
from typing import Any, Dict, List
from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import os, re, json, urllib.parse, http.client
from datetime import datetime, timezone

from .chat_memory import ChatMemory

STOP_SEQS = ["\nUser:", "\nAssistant:", "\nSystem:", "User:", "Assistant:", "SYSTEM:", "USER:", "ASSISTANT:"]

def _clean_assistant_reply(text: str) -> str:
    if not isinstance(text, str):
        return text
    m = re.search(r'\n(?:User|Assistant|System)\s*:', text, flags=re.IGNORECASE)
    if m:
        text = text[:m.start()]
    text = re.sub(r'^(?:User|Assistant|System)\s*:\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def _tail_text(path: str, max_bytes: int = 20_000, max_lines: int = 60) -> str:
    try:
        if not os.path.exists(path):
            return ""
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                data = f.read()
            else:
                data = f.read()
        text = data.decode(errors="ignore")
        lines = text.splitlines()
        def keep(line: str) -> bool:
            s = line.strip()
            if not s: return False
            if any(tag in s for tag in ("ERROR","INFO","WARN","WARNING","Scheduler","Strategy","Universe","RiskManager","ExecutionEngine","Portfolio")):
                return True
            letters = sum(ch.isalpha() for ch in s)
            return (letters / max(len(s),1)) >= 0.6
        filtered = [ln for ln in lines if keep(ln)]
        cleaned, ts_re, num_token = [], re.compile(r"\b\d{2,4}[-/:]\d{1,2}[-/:]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b"), re.compile(r"\b\d[\d:\-\.]*\b")
        for ln in filtered[-max_lines:]:
            s = ts_re.sub("", ln); s = num_token.sub("", s); s = re.sub(r"\s{2,}", " ", s).strip()
            if s: cleaned.append(s)
        return "\n".join(cleaned)
    except Exception:
        return ""

def _load_persona(default_text: str) -> str:
    p = Path("config/ai-persona.md")
    try:
        if p.exists(): return p.read_text(encoding="utf-8").strip()
    except Exception: pass
    return default_text.strip()

def _extract_facts(user_text: str) -> Dict[str, str]:
    facts: Dict[str, str] = {}
    if not user_text: return facts
    m = re.search(r"(?:my\s+favorite\s+color\s+is|i\s+like\s+the\s+color)\s+([A-Za-z]+)", user_text, flags=re.IGNORECASE)
    if m: facts["favorite_color"] = m.group(1).lower()
    return facts

def _profile_system(facts: Dict[str, Any]) -> str:
    parts = []
    if "favorite_color" in facts: parts.append(f"favorite_color={facts['favorite_color']}")
    return "Session profile (persisted facts): " + ", ".join(parts) if parts else ""

def _market_grounding(controller: Any) -> str:
    try:
        bot = getattr(controller, "_bot", None)
        alpaca = getattr(bot, "alpaca", None) if bot else None
        if alpaca and hasattr(alpaca, "get_clock"):
            clk = alpaca.get_clock()
            is_open = bool(getattr(clk, "is_open", None))
            next_open = getattr(clk, "next_open", None)
            next_close = getattr(clk, "next_close", None)
            status = "OPEN" if is_open else "CLOSED"
            msg = f"Market status: {status}."
            if next_open: msg += f" Next open: {next_open}"
            if next_close: msg += f" Next close: {next_close}"
            return msg
        sched = getattr(bot, "scheduler", None) if bot else None
        if sched and hasattr(sched, "is_market_open"):
            status = "OPEN" if bool(sched.is_market_open()) else "CLOSED"
            return f"Market status (scheduler): {status}."
    except Exception:
        pass
    return "Market status unknown from backend; do not assume today's status."

def _http_get(host: str, path: str, headers: Dict[str, str], use_https: bool = True, timeout: int = 10):
    conn = (http.client.HTTPSConnection if use_https else http.client.HTTPConnection)(host, timeout=timeout)
    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, resp.reason, data
    finally:
        conn.close()

def _http_get_json(host: str, path: str, headers: Dict[str, str]) -> Dict[str, Any]:
    status, reason, data = _http_get(host, path, headers, use_https=True)
    if status != 200:
        raise RuntimeError(f"{host} HTTP {status} {reason}")
    return json.loads(data.decode("utf-8", errors="ignore"))

def _web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    provider = (os.getenv("SEARCH_PROVIDER") or "").lower()
    q = urllib.parse.quote(query.strip())
    results: List[Dict[str, str]] = []
    try:
        if provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY"); cse_id = os.getenv("GOOGLE_CSE_ID")
            if not api_key or not cse_id:
                return {"error": "Google search not configured (set GOOGLE_API_KEY and GOOGLE_CSE_ID)."}
            data = _http_get_json("www.googleapis.com", f"/customsearch/v1?q={q}&key={api_key}&cx={cse_id}&num={max_results}", headers={})
            for item in data.get("items", [])[:max_results]:
                results.append({"title": item.get("title",""), "url": item.get("link",""), "snippet": item.get("snippet","")})

        elif provider == "bing":
            key = os.getenv("BING_API_KEY")
            if not key: return {"error": "Bing search not configured (set BING_API_KEY)."}
            data = _http_get_json("api.bing.microsoft.com", f"/v7.0/search?q={q}&count={max_results}", headers={"Ocp-Apim-Subscription-Key": key})
            for item in (data.get("webPages", {}) or {}).get("value", [])[:max_results]:
                results.append({"title": item.get("name",""), "url": item.get("url",""), "snippet": item.get("snippet","")})

        elif provider == "brave":
            key = os.getenv("BRAVE_API_KEY")
            if not key: return {"error": "Brave search not configured (set BRAVE_API_KEY)."}
            data = _http_get_json("api.search.brave.com", f"/res/v1/web/search?q={q}&count={max_results}", headers={"X-Subscription-Token": key})
            for item in (data.get("web", {}) or {}).get("results", [])[:max_results]:
                results.append({"title": item.get("title",""), "url": item.get("url",""), "snippet": item.get("description","")})

        elif provider == "serpapi":
            key = os.getenv("SERPAPI_KEY")
            if not key: return {"error": "SerpAPI not configured (set SERPAPI_KEY)."}
            data = _http_get_json("serpapi.com", f"/search.json?q={q}&engine=google&num={max_results}&api_key={key}", headers={})
            for item in (data.get("organic_results", []) or [])[:max_results]:
                results.append({"title": item.get("title",""), "url": item.get("link",""), "snippet": item.get("snippet","")})

        else:
            return {"error": "Web search not configured. Set SEARCH_PROVIDER and API key(s)."}
    except Exception as e:
        return {"error": f"Web search failed: {e}"}
    return {"provider": provider, "results": results}

def _fetch_page_text(url: str = None) -> str:
    """
    Fetch a web page and return cleaned text.
    Tries HTTPS first; falls back to HTTP if needed. Light HTML cleanup.
    """
    try:
        if not url: return ""
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.org)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        status, _, data = _http_get(host, path, headers, use_https=(parsed.scheme=="https"))
        if status != 200:
            status2, _, data2 = _http_get(host, path, headers, use_https=False)
            if status2 == 200:
                data = data2
            else:
                return ""
        html = data.decode("utf-8", errors="ignore")
        text = html
        try:
            from bs4 import BeautifulSoup  # optional
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script","style","noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except Exception:
            text = re.sub(r"(?is)<script.*?>.*?</script>", "", html)
            text = re.sub(r"(?is)<style.*?>.*?</style>", "", text)
            text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", "", text)
            text = re.sub(r"(?is)<.*?>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = text.strip()
        if len(text) > 12000:
            text = text[:12000] + "\n...[truncated]..."
        return text
    except Exception:
        return ""

def _summarize_snippets(snippets: List[str], limit: int = 4) -> List[str]:
    out = []
    for s in snippets:
        s = (s or "").strip()
        if not s:
            continue
        parts = re.split(r"(?<=[.!?])\s+", s)
        out.append(" ".join(parts[:2]))
        if len(out) >= limit:
            break
    return out

def _run_web_research(query: str, max_results: int = 6, max_fetch: int = 3) -> str:
    """
    Full web research: search + fetch top pages + short digest + citations.
    """
    search = _web_search(query, max_results=max_results)
    if "results" not in search or not search["results"]:
        err = search.get("error") or "no results"
        return f"Web research: unavailable ({err})."
    results = search["results"]
    snippets = [r.get("snippet","") for r in results]
    snippet_summary = _summarize_snippets(snippets, limit=4)
    digests = []
    citations = []
    for i, r in enumerate(results[:max_fetch], 1):
        url = r.get("url","")
        title = (r.get("title","") or "").strip()
        page_text = _fetch_page_text(url)
        if page_text:
            extract = page_text[:1200]
            sentences = re.split(r"(?<=[.!?])\s+", extract)
            extract_clean = " ".join(sentences[:5]).strip()
            digests.append(f"- {title}: {extract_clean}")
            citations.append(f"{i}. {title} — {url}")
        else:
            digests.append(f"- {title}: [content unavailable]")
            citations.append(f"{i}. {title} — {url}")
    lines = []
    lines.append("Web research digest:")
    if snippet_summary:
        lines.append("Key points from search snippets:")
        for ss in snippet_summary:
            lines.append(f"• {ss}")
    if digests:
        lines.append("Top sources (page extracts):")
        for d in digests:
            lines.append(d)
    if citations:
        lines.append("Citations:")
        for c in citations:
            lines.append(c)
    return "\n".join(lines)

def _fetch_quote(symbol: str) -> Dict[str, Any]:
    sym = urllib.parse.quote(symbol.upper())
    try:
        data = _http_get_json("query1.finance.yahoo.com", f"/v7/finance/quote?symbols={sym}", headers={})
        res = (data.get("quoteResponse", {}) or {}).get("result", []) or []
        if not res:
            return {"error": "No quote data returned"}
        q = res[0]
        price = q.get("regularMarketPrice")
        change = q.get("regularMarketChange")
        pct = q.get("regularMarketChangePercent")
        state = q.get("marketState") or "UNKNOWN"
        ts = q.get("regularMarketTime")
        as_of = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None
        currency = q.get("currency") or "USD"
        return {
            "symbol": symbol.upper(),
            "price": price, "change": change, "change_pct": pct,
            "market_state": state, "as_of_utc": as_of, "currency": currency,
            "source": "Yahoo Finance",
        }
    except Exception as e:
        return {"error": f"Quote fetch failed: {e}"}

PRICE_PATTERNS = [r"\bcurrent\s+price\b", r"\bprice\s+(?:now|right\s+now|currently)\b", r"\bwhat'?s\s+the\s+current\s+.*price\b", r"\bquote\b"]
def _extract_price_symbol(text: str) -> str:
    s = (text or "").strip()
    if not s: return ""
    if any(re.search(p, s, flags=re.IGNORECASE) for p in PRICE_PATTERNS) or re.search(r"\bprice\b", s, flags=re.IGNORECASE):
        m = re.search(r"\b([A-Z]{1,5})\b", s)
        if m: return m.group(1)
    return ""

def make_router(runtime: Any, registry: Any, memory: ChatMemory, logs_path: str = "logs/integrated.log", controller: Any = None):
    router = APIRouter()

    @router.get("/search/status")
    async def search_status():
        provider = (os.getenv("SEARCH_PROVIDER") or "").lower() or "none"
        return {
            "provider": provider,
            "google_key_set": bool(os.getenv("GOOGLE_API_KEY")),
            "google_cx_set": bool(os.getenv("GOOGLE_CSE_ID")),
            "bing_key_set": bool(os.getenv("BING_API_KEY")),
            "brave_key_set": bool(os.getenv("BRAVE_API_KEY")),
            "serpapi_key_set": bool(os.getenv("SERPAPI_KEY")),
        }

    @router.get("/search/web")
    async def search_web(q: str = Query(...), max_results: int = Query(5)):
        try:
            data = _web_search(q, max_results=max_results)
            return data
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.post("/ui/chat/open")
    async def open_chat(payload: Dict[str, Any] = Body(None)):
        try:
            default_prompt = (
                "You are a friendly, professional trading assistant. Be honest (never fabricate), concise, and keep continuity. "
                "Use logs only as context; never echo raw numbers or reproduce logs. "
                "Do not write lines beginning with 'User:' or 'Assistant:'—only answer as yourself. "
                "If a question needs real-time info and you cannot verify, say so plainly and offer a next step (check the feed or run web research). "
                "If web research is provided, use it and cite sources briefly (title + URL)."
            )
            persona = _load_persona(default_prompt)
            if "Do not write lines beginning with 'User:'" not in persona:
                persona += "\n\nDo not write lines beginning with 'User:' or 'Assistant:'. Only answer as yourself."
            sid = memory.open((payload or {}).get("system") or persona)
            return {"session_id": sid}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/ui/chat/history")
    async def history(session_id: str):
        try:
            return {"session_id": session_id, "messages": memory.messages(session_id)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.post("/ui/chat/send")
    async def send(payload: Dict[str, Any] = Body(...)):
        try:
            sid = payload.get("session_id")
            content = (payload.get("content") or "").strip()
            include_logs = bool(payload.get("include_logs", False))

            provider_configured = bool(os.getenv("SEARCH_PROVIDER"))
            include_web = bool(payload.get("include_web", False) or provider_configured)

            tokens = int(payload.get("tokens") or 256)
            if not sid or not content:
                return JSONResponse(status_code=400, content={"error": "Missing session_id or content"})

            memory.add_user(sid, content)
            for k, v in _extract_facts(content).items():
                memory.set_fact(sid, k, v)

            facts = memory.get_facts(sid)
            prof = _profile_system(facts)
            if prof: memory.inject_system(sid, prof)
            if controller is not None:
                memory.inject_system(sid, _market_grounding(controller))

            if include_logs:
                tail = _tail_text(logs_path)
                if tail:
                    memory.inject_system(sid, "Recent ops context (reference only; do not quote numbers):\n" + tail)

            symbol = _extract_price_symbol(content)
            if symbol:
                quote = _fetch_quote(symbol)
                if "error" in quote:
                    memory.inject_system(sid, f"Live quote unavailable for {symbol}: {quote['error']}")
                else:
                    price = quote.get("price"); change = quote.get("change"); pct = quote.get("change_pct")
                    state = quote.get("market_state"); as_of = quote.get("as_of_utc"); src = quote.get("source")
                    memory.inject_system(sid, f"Live quote | {symbol} price={price} change={change} ({pct}%) state={state} as_of={as_of} UTC | Source: {src}")

            if include_web:
                research = _run_web_research(content, max_results=6, max_fetch=3)
                memory.inject_system(sid, research)

            alias = getattr(registry, "get_default_alias", lambda: None)()
            messages = memory.messages(sid)
            options = {
                "num_predict": tokens,
                "temperature": 0.32,
                "top_p": 0.9,
                "repeat_penalty": 1.12,
                "stop": STOP_SEQS,
            }
            text = runtime.chat(alias=alias, messages=messages, options=options, stream=False)
            text = _clean_assistant_reply(text)

            memory.add_assistant(sid, text)
            return {"session_id": sid, "reply": text}
        except Exception as e:
            alias = getattr(registry, "get_default_alias", lambda: None)()
            return JSONResponse(status_code=500, content={"error": str(e), "alias": alias})

    return router