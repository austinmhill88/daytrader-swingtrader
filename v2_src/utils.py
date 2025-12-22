import numpy as np
import pandas as pd

def zscore(series: pd.Series, window: int = 20):
    s = (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)
    return s

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(window).mean()

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()