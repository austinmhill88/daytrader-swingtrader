import math
from src.models import OrderIntent
from loguru import logger

class ExecutionEngine:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg

    def size_for_signal(self, equity: float, price: float, atr: float, per_trade_risk_pct: float) -> int:
        risk_amount = equity * per_trade_risk_pct / 100.0
        stop_distance = max(atr, price * 0.005)  # fallback stop distance
        qty = math.floor(risk_amount / stop_distance)
        return max(qty, 0)

    def place_intents(self, intents):
        for intent in intents:
            try:
                order = self.client.place_order(
                    symbol=intent.symbol,
                    qty=intent.qty,
                    side=intent.side,
                    order_type=intent.order_type,
                    time_in_force=intent.time_in_force,
                    limit_price=intent.limit_price,
                    stop_loss=intent.bracket.get("stop_loss") if intent.bracket else None,
                    take_profit=intent.bracket.get("take_profit") if intent.bracket else None
                )
                logger.info(f"Placed order: {order}")
            except Exception as e:
                logger.error(f"Execution error: {e}")