import alpaca_trade_api as tradeapi
from loguru import logger

class AlpacaClient:
    def __init__(self, key_id: str, secret_key: str, base_url: str):
        self.api = tradeapi.REST(key_id, secret_key, base_url)

    def account(self):
        return self.api.get_account()

    def positions(self):
        return self.api.list_positions()

    def place_order(self, symbol, qty, side, order_type="limit", time_in_force="day",
                    limit_price=None, stop_loss=None, take_profit=None):
        params = dict(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        if limit_price is not None:
            params["limit_price"] = str(limit_price)
        if stop_loss or take_profit:
            params["order_class"] = "bracket"
            if take_profit:
                params["take_profit"] = {"limit_price": str(take_profit)}
            if stop_loss:
                params["stop_loss"] = {"stop_price": str(stop_loss)}
        try:
            order = self.api.submit_order(**params)
            logger.info(f"Order submitted: {order}")
            return order
        except Exception as e:
            logger.error(f"Order error for {symbol}: {e}")
            raise

    def cancel_all(self):
        self.api.cancel_all_orders()

    def get_latest_bar(self, symbol: str):
        # Using v2 data API via REST (assuming subscription)
        bars = self.api.get_bars(symbol, "1Min", limit=1)
        return bars[0] if bars else None