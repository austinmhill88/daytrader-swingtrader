from typing import List
from loguru import logger
from src.models import OrderIntent

class RiskManager:
    def __init__(self, cfg, portfolio_state):
        self.cfg = cfg
        self.portfolio = portfolio_state  # abstraction to positions, equity, exposure

    def pre_trade_checks(self, intents: List[OrderIntent]) -> List[OrderIntent]:
        safe_intents = []
        for intent in intents:
            if self.portfolio.daily_drawdown_pct() >= self.cfg["daily_max_drawdown_pct"]:
                logger.error("Daily drawdown breached; kill-switch engaged.")
                continue
            if self.portfolio.projected_gross_exposure_pct(intent) > self.cfg["max_gross_exposure_pct_intraday"]:
                logger.warning("Exposure cap would be breached; skipping intent.")
                continue
            safe_intents.append(intent)
        return safe_intents