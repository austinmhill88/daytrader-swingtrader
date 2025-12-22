from typing import Dict
from loguru import logger

class PortfolioState:
    def __init__(self, client):
        self.client = client

    def equity(self) -> float:
        acct = self.client.account()
        return float(acct.equity)

    def positions(self):
        return self.client.positions()

    def daily_drawdown_pct(self) -> float:
        acct = self.client.account()
        # Simple proxy: (equity - last_equity_high) / last_equity_high * 100
        # For demo purposes, use portfolio intraday PnL if available
        try:
            dd = float(acct.daytrading_buying_power)  # placeholder; replace with actual dd tracking
        except Exception:
            dd = 0.0
        return dd

    def projected_gross_exposure_pct(self, intent) -> float:
        # Placeholder: compute gross exposure after adding intent
        return 10.0