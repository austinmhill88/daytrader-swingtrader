import numpy as np
import pandas as pd
from typing import List
from src.models import Signal
from src.utils import zscore, atr, ema

class IntradayMeanReversion:
    name = "intraday_mean_reversion"

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {}  # symbol -> DataFrame rolling

    def warmup(self, historical_df: pd.DataFrame):
        # Expect dataframe with columns: symbol, ts, open, high, low, close, volume, vwap
        for sym, df in historical_df.groupby("symbol"):
            df = df.sort_values("ts")
            df["vw"] = df["vwap"].fillna((df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum())
            df["z_vw"] = zscore(df["close"] - df["vw"], window=20)
            df["atr1"] = atr(df["high"], df["low"], df["close"], window=14)
            self.state[sym] = df.iloc[-300:].copy()

    def on_bar(self, bar) -> List[Signal]:
        sym = bar.S
        df = self.state.get(sym)
        if df is None:
            return []
        # Append and recompute minimal features
        new = {
            "ts": bar.t,
            "open": bar.o,
            "high": bar.h,
            "low": bar.l,
            "close": bar.c,
            "volume": bar.v,
            "vwap": getattr(bar, "vw", None)
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df["vw"] = df["vwap"].fillna((df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum())
        df["z_vw"] = zscore(df["close"] - df["vw"], window=20)
        df["atr1"] = atr(df["high"], df["low"], df["close"], window=14)

        self.state[sym] = df.iloc[-600:]  # keep window

        z = float(df["z_vw"].iloc[-1])
        atr_now = float(df["atr1"].iloc[-1])
        strength = 0.0
        reason = ""

        if z <= -self.cfg["zscore_entry"]:
            strength = +1.0
            reason = f"zscore {z:.2f} <= -{self.cfg['zscore_entry']}"
        elif z >= self.cfg["zscore_entry"]:
            strength = -1.0
            reason = f"zscore {z:.2f} >= {self.cfg['zscore_entry']}"

        if strength == 0.0 or np.isnan(atr_now) or atr_now <= 0:
            return []

        confidence = min(1.0, abs(z) / (self.cfg["zscore_entry"] + 0.5))
        sig = Signal(symbol=sym, strength=strength, confidence=confidence, reason=reason)
        return [sig]

    def on_end_of_day(self):
        pass