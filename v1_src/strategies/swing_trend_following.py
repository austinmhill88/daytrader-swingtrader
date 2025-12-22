import pandas as pd
from typing import List
from src.models import Signal
from src.utils import ema

class SwingTrendFollowing:
    name = "swing_trend_following"

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {}

    def warmup(self, historical_df: pd.DataFrame):
        for sym, df in historical_df.groupby("symbol"):
            df = df.sort_values("ts")
            df["ema_fast"] = ema(df["close"], self.cfg["ema_fast"])
            df["ema_slow"] = ema(df["close"], self.cfg["ema_slow"])
            df["trend"] = (df["ema_fast"] > df["ema_slow"]).astype(int) - (df["ema_fast"] < df["ema_slow"]).astype(int)
            self.state[sym] = df.iloc[-500:].copy()

    def on_bar(self, bar) -> List[Signal]:
        sym = bar.S
        df = self.state.get(sym)
        if df is None:
            return []
        new = {"ts": bar.t, "close": bar.c}
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df["ema_fast"] = ema(df["close"], self.cfg["ema_fast"])
        df["ema_slow"] = ema(df["close"], self.cfg["ema_slow"])
        df["trend"] = (df["ema_fast"] > df["ema_slow"]).astype(int) - (df["ema_fast"] < df["ema_slow"]).astype(int)
        self.state[sym] = df.iloc[-600:]

        recent_trend = int(df["trend"].iloc[-1])
        if recent_trend > 0:
            return [Signal(symbol=sym, strength=+0.5, confidence=0.6, reason="Uptrend")]
        elif recent_trend < 0:
            return [Signal(symbol=sym, strength=-0.5, confidence=0.6, reason="Downtrend")]
        return []