import threading
import time
from typing import Optional, Dict, Any

from loguru import logger
from src.main import TradingBot

class TradingController:
    """
    Controls the TradingBot lifecycle inside the FastAPI server.
    Runs TradingBot.start() in a background thread; supports stop/status.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._bot: Optional[TradingBot] = None
        self._thr: Optional[threading.Thread] = None
        self._state: Dict[str, Any] = {
            "running": False,
            "start_time": None,
            "stop_time": None,
            "last_error": None,
        }

    def start(self) -> Dict[str, Any]:
        if self._state["running"]:
            return {"ok": True, "message": "Already running"}
        try:
            self._bot = TradingBot(self.config_path)
        except Exception as e:
            self._state["last_error"] = str(e)
            logger.exception(e)
            return {"ok": False, "error": f"Init failed: {e}"}

        def _run():
            try:
                self._state["running"] = True
                self._state["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                # Blocking run; returns when bot.stop() is called
                self._bot.start()
            except Exception as e:
                logger.exception(e)
                self._state["last_error"] = str(e)
            finally:
                self._state["running"] = False
                self._state["stop_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Daemon thread so process can exit if needed
        self._thr = threading.Thread(target=_run, name="TradingBotThread", daemon=True)
        self._thr.start()
        return {"ok": True, "message": "Started"}

    def stop(self) -> Dict[str, Any]:
        # Idempotent stop
        try:
            if self._bot:
                self._bot.stop()
        except Exception as e:
            logger.exception(e)
            return {"ok": False, "error": f"Stop failed: {e}"}

        # Best-effort join so shutdown is quick
        try:
            if self._thr and self._thr.is_alive():
                self._thr.join(timeout=5)
        except Exception as e:
            logger.warning(f"Join failed: {e}")

        self._state["running"] = False
        self._state["stop_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return {"ok": True, "message": "Stopped"}

    def status(self) -> Dict[str, Any]:
        return dict(self._state)