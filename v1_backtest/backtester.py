import argparse
import pandas as pd
from loguru import logger

def simulate_intraday(strategy, data_by_symbol, cfg):
    # Event-driven loop over bars per symbol
    pnl = 0.0
    for sym, df in data_by_symbol.items():
        strategy.warmup(df.iloc[:1000])
        positions = 0
        for i in range(1000, len(df)):
            bar = df.iloc[i]
            signals = strategy.on_bar(type("Bar", (), {"S": sym, "t": bar["ts"], "o": bar["open"], "h": bar["high"], "l": bar["low"], "c": bar["close"], "v": bar["volume"], "vw": bar.get("vwap", None)}))
            for sig in signals:
                # naive fill simulation
                if sig.strength > 0:
                    positions += 100
                    pnl -= 100 * bar["close"]
                else:
                    positions -= 100
                    pnl += 100 * bar["close"]
        logger.info(f"{sym} PnL: {pnl}")
    return pnl

def main(config_path):
    logger.info(f"Backtesting with config {config_path}")
    # Load historical data (placeholder)
    # data_by_symbol = {sym: df}
    # strategy = IntradayMeanReversion(cfg)
    # res = simulate_intraday(strategy, data_by_symbol, cfg)
    # logger.info(f"Total PnL: {res}")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)