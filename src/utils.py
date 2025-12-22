"""
Utility functions for technical analysis and data processing.
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling z-score of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Z-score series
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    # Add small epsilon to avoid division by zero
    return (series - mean) / (std + 1e-9)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Period for ATR calculation
        
    Returns:
        ATR series
    """
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        series: Input series
        span: EMA span/period
        
    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        series: Input series
        window: Window size
        
    Returns:
        SMA series
    """
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        series: Price series
        period: RSI period
        
    Returns:
        RSI series (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-9)
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        series: Price series
        window: Window for moving average
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
        
    Returns:
        ADX series
        
    Raises:
        ValueError: If series lengths don't match or contain invalid values
    """
    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close series must have the same length")
    
    if high.isna().any() or low.isna().any() or close.isna().any():
        raise ValueError("Input series contain NaN values")
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # Calculate ATR
    atr_values = atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_values)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_values)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx_values = dx.rolling(period).mean()
    
    return adx_values


def vwap(close: pd.Series, volume: pd.Series, high: pd.Series = None, 
         low: pd.Series = None) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        close: Close prices
        volume: Volume
        high: High prices (optional, for HLC/3)
        low: Low prices (optional, for HLC/3)
        
    Returns:
        VWAP series
    """
    if high is not None and low is not None:
        typical_price = (high + low + close) / 3
    else:
        typical_price = close
    
    return (typical_price * volume).cumsum() / volume.cumsum()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_period).mean()
    
    return k, d


def calculate_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns over specified periods.
    
    Args:
        series: Price series
        periods: Number of periods for returns
        
    Returns:
        Returns series
    """
    return series.pct_change(periods=periods)


def calculate_volatility(series: pd.Series, window: int = 20, annualize: bool = False) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        series: Price series
        window: Window for volatility calculation
        annualize: Whether to annualize (assuming daily data)
        
    Returns:
        Volatility series
    """
    returns = calculate_returns(series)
    vol = returns.rolling(window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)  # Assuming 252 trading days
    
    return vol


def safe_divide(numerator: Union[float, pd.Series], 
                denominator: Union[float, pd.Series],
                fill_value: float = 0.0) -> Union[float, pd.Series]:
    """
    Safely divide, handling division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        fill_value: Value to use when denominator is zero
        
    Returns:
        Division result
    """
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(fill_value)
    else:
        return numerator / denominator if denominator != 0 else fill_value


def normalize_series(series: pd.Series, min_val: float = 0.0, max_val: float = 1.0) -> pd.Series:
    """
    Normalize series to specified range.
    
    Args:
        series: Input series
        min_val: Minimum value of normalized range
        max_val: Maximum value of normalized range
        
    Returns:
        Normalized series
    """
    series_min = series.min()
    series_max = series.max()
    
    if series_max == series_min:
        return pd.Series([(min_val + max_val) / 2] * len(series), index=series.index)
    
    normalized = (series - series_min) / (series_max - series_min)
    return normalized * (max_val - min_val) + min_val
