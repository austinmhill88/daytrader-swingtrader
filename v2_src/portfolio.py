class PortfolioState:
    def __init__(self, client):
        self.client = client
        self._equity_high = None
        self._equity_start = None

    def equity(self) -> float:
        acct = self.client.account()
        eq = float(acct.equity)
        if self._equity_start is None:
            self._equity_start = eq
        self._equity_high = max(self._equity_high or eq, eq)
        return eq

    def positions(self):
        return self.client.positions()

    def daily_drawdown_pct(self) -> float:
        if self._equity_high is None or self._equity_high == 0:
            return 0.0
        eq = self.equity()
        dd = (self._equity_high - eq) / self._equity_high * 100.0
        return max(dd, 0.0)

    def projected_gross_exposure_pct(self, intent) -> float:
        # Placeholder: compute gross exposure after adding intent
        # Improve with actual position and notional tracking
        return 10.0