"""
AI Ops routes: Compose logs/status context and ask the AI for an insight summary.
Chunked summarization to keep prompts within the model's context window.
"""
import os
import io
import time
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from .model_runtime import AIModelRuntime, ModelRegistry

# Hard caps to keep within context window
MAX_CHARS_PER_LOG = 6000   # ~2-3k tokens depending on text
FINAL_SUMMARY_TOKENS = 384
PER_LOG_SUMMARY_TOKENS = 256

def make_router(registry: ModelRegistry, runtime: AIModelRuntime, default_alias: Optional[str]):
    router = APIRouter()

    def _read_tail(path: str, lines: int = 120) -> str:
        try:
            if not os.path.exists(path):
                return ""
            with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.readlines()[-lines:]
                text = "".join(data)
                # Trim to cap
                return text[-MAX_CHARS_PER_LOG:]
        except Exception:
            return ""

    def _summarize(alias: Optional[str], title: str, text: str, tokens: int) -> str:
        if not text.strip():
            return f"{title}: No recent entries."
        prompt = (
            f"Summarize the following log segment ({title}) for a trading operations dashboard.\n"
            "Use trader lingo, concise bullets: signals/orders, risk/exposure/P&L, errors/alerts, next actions.\n\n"
            f"== {title} ==\n{text}\n"
        )
        try:
            return runtime.generate(alias or default_alias, prompt, options={"num_predict": tokens}, stream=False).strip()
        except Exception as e:
            return f"{title}: Summary error: {e}"

    @router.post("/ai/insight")
    async def insight(
        lines: int = Query(120, description="Lines to read from each log (keep modest to avoid context overflow)")
    ):
        """
        Summarize recent runtime/errors/trades logs and give next actions.
        Chunked to avoid context overflow.
        """
        runtime_log = _read_tail("logs/runtime.log", lines)
        errors_log = _read_tail("logs/errors.log", lines)
        trades_log = _read_tail("logs/trades.log", lines)

        if not (runtime_log or errors_log or trades_log):
            return JSONResponse(status_code=404, content={"error": "No logs found"})

        # First-level summaries (smaller prompts)
        runtime_summary = _summarize(default_alias, "RUNTIME LOG", runtime_log, PER_LOG_SUMMARY_TOKENS)
        errors_summary = _summarize(default_alias, "ERRORS LOG", errors_log, PER_LOG_SUMMARY_TOKENS)
        trades_summary = _summarize(default_alias, "TRADES LOG", trades_log, PER_LOG_SUMMARY_TOKENS)

        # Final insight composed from summaries (small prompt)
        final_prompt = (
            "You are a pragmatic trading floor assistant.\n"
            "Combine the summaries into a concise operator insight with bullets first, then rationale.\n"
            "Sections: Status summary; Signals/orders; Risk/exposure/P&L; Errors/alerts; Next actions.\n\n"
            f"{runtime_summary}\n\n{errors_summary}\n\n{trades_summary}\n"
        )
        try:
            text = runtime.generate(default_alias, final_prompt, options={"num_predict": FINAL_SUMMARY_TOKENS}, stream=False)
            return {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "insight": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router