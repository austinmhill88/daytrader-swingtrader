import argparse
import os
from loguru import logger
from src.config import load_config
from src.logging_utils import setup_logging
from src.alpaca_client import AlpacaClient
from src.portfolio import PortfolioState
from src.execution_engine import ExecutionEngine
from src.risk_manager import RiskManager
from src.strategies.intraday_mean_reversion import IntradayMeanReversion
from src.strategies.swing_trend_following import SwingTrendFollowing

def build_universe(client, cfg):
    # Placeholder: filter tickers by ADV, spread using stored analytics or external source
    # For demo: pick a small set
    return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

def run(cfg_path: str, live: bool):
    cfg = load_config(cfg_path)
    logger = setup_logging(cfg["storage"]["logs_dir"])

    client = AlpacaClient(cfg["alpaca"]["key_id"], cfg["alpaca"]["secret_key"], cfg["alpaca"]["base_url"])
    portfolio = PortfolioState(client)
    exec_engine = ExecutionEngine(client, cfg["execution"])
    risk_mgr = RiskManager(cfg["risk"], portfolio)

    universe = build_universe(client, cfg)
    logger.info(f"Universe: {universe}")

    # Warm-up strategies with historical data (placeholder)
    mr = IntradayMeanReversion(cfg["strategies"]["intraday_mean_reversion"])
    swing = SwingTrendFollowing(cfg["strategies"]["swing_trend_following"])
    # TODO: fetch historical_df per symbol
    # mr.warmup(historical_df)
    # swing.warmup(historical_df)

    # Live data loop placeholder
    for sym in universe:
        bar = client.get_latest_bar(sym)
        if bar is None:
            continue
        # signals
        mr_signals = []  # mr.on_bar(bar)
        swing_signals = []  # swing.on_bar(bar)
        signals = mr_signals + swing_signals

        intents = []
        for sig in signals:
            price = bar.c
            # placeholder atr
            atr_now = max(0.5, price * 0.01)
            qty = exec_engine.size_for_signal(portfolio.equity(), price, atr_now, cfg["risk"]["per_trade_risk_pct"])
            if qty <= 0:
                continue
            side = "buy" if sig.strength > 0 else "sell"
            limit_price = price * (1 - 0.0005) if side == "buy" else price * (1 + 0.0005)
            bracket = {}
            if cfg["execution"]["use_bracket_orders"]:
                stop = price - atr_now if side == "buy" else price + atr_now
                tp = price + atr_now * 1.5 if side == "buy" else price - atr_now * 1.5
                bracket = {"stop_loss": stop, "take_profit": tp}
            intents.append({
                "symbol": sig.symbol,
                "qty": qty,
                "side": side,
                "order_type": cfg["execution"]["default_order_type"],
                "time_in_force": cfg["execution"]["time_in_force"],
                "limit_price": round(limit_price, 2),
                "bracket": bracket
            })
        # convert dicts to OrderIntent and pass through risk
        from src.models import OrderIntent
        intents = [OrderIntent(**i) for i in intents]
        safe_intents = risk_mgr.pre_trade_checks(intents)
        exec_engine.place_intents(safe_intents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()
    run(args.config, live=args.live)