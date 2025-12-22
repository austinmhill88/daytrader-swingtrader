import asyncio
from loguru import logger
from typing import Callable, List
import alpaca_trade_api as tradeapi

class LiveDataFeed:
    def __init__(self, key_id, secret_key, base_url, symbols: List[str]):
        self.conn = tradeapi.Stream(key_id, secret_key, base_url)
        self.symbols = symbols

    def subscribe_bars(self, callback: Callable):
        async def on_bar(bar):
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")

        for sym in self.symbols:
            self.conn.subscribe_bars(on_bar, sym)

    def run(self):
        try:
            self.conn.run()
        except Exception as e:
            logger.error(f"Data stream error: {e}")
            # TODO: backoff and reconnect

class HistoricalDataFeed:
    def __init__(self, rest: tradeapi.REST):
        self.rest = rest

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str, limit: int = 10000):
        return self.rest.get_bars(symbol, timeframe, start=start, end=end, limit=limit)