"""
Intraday mean reversion strategy based on z-score and VWAP.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime, time

from src.strategies.base import Strategy
from src.models import Signal, Bar
from src.utils import zscore, atr, ema, rsi
from loguru import logger


class IntradayMeanReversion(Strategy):
    """
    Intraday mean reversion strategy using z-score relative to VWAP.
    
    Entry conditions:
    - Z-score < -threshold (oversold) for long
    - Z-score > +threshold (overbought) for short
    - Sufficient volume and liquidity
    - Within allowed trading hours
    
    Exit conditions:
    - Z-score returns toward mean
    - ATR-based stop loss
    - End of day flatten
    """
    
    name = "intraday_mean_reversion"
    
    def __init__(self, config: Dict):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Strategy parameters
        self.zscore_entry = config.get('zscore_entry', 2.0)
        self.zscore_exit = config.get('zscore_exit', 0.5)
        self.atr_stop_multiplier = config.get('atr_stop_multiplier', 1.5)
        self.max_positions = config.get('max_positions', 20)
        self.min_volume = config.get('min_volume', 100000)
        
        # Time filters
        time_filters = config.get('time_filters', {})
        self.skip_first_minutes = time_filters.get('skip_first_minutes', 3)
        self.reduce_last_minutes = time_filters.get('reduce_last_minutes', 5)
        self.flatten_eod = time_filters.get('flatten_eod', True)
        
        # Lookback periods
        lookback = config.get('lookback_periods', {})
        self.zscore_window = lookback.get('zscore_window', 20)
        self.atr_window = lookback.get('atr_window', 14)
        self.vwap_window = lookback.get('vwap_window', 20)
        
        logger.info(
            f"{self.name} initialized | "
            f"zscore_entry={self.zscore_entry}, "
            f"atr_stop_mult={self.atr_stop_multiplier}"
        )
    
    def warmup(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Warm up with historical data.
        
        Args:
            historical_data: Dictionary mapping symbol to DataFrame
        """
        logger.info(f"{self.name}: Warming up with data for {len(historical_data)} symbols")
        
        for symbol, df in historical_data.items():
            if df is None or len(df) == 0:
                continue
            
            try:
                # Calculate indicators
                df = self._calculate_indicators(df)
                
                # Store limited history
                self._update_state(symbol, df.iloc[-300:].copy())
                
            except Exception as e:
                logger.error(f"{self.name}: Error warming up {symbol}: {e}")
        
        self.is_warmed_up = True
        logger.info(f"{self.name}: Warmup complete for {len(self.state)} symbols")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # VWAP calculation (use provided or calculate)
        if 'vwap' not in df.columns or df['vwap'].isna().all():
            df['vwap_calc'] = (df['close'] * df['volume']).rolling(self.vwap_window).sum() / \
                             df['volume'].rolling(self.vwap_window).sum()
        else:
            df['vwap_calc'] = df['vwap']
        
        # Z-score relative to VWAP
        price_diff = df['close'] - df['vwap_calc']
        df['zscore_vwap'] = zscore(price_diff, window=self.zscore_window)
        
        # ATR for stops
        df['atr'] = atr(df['high'], df['low'], df['close'], window=self.atr_window)
        
        # Additional filters
        df['rsi'] = rsi(df['close'], period=7)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df
    
    def _is_trading_time(self) -> bool:
        """
        Check if current time is within allowed trading hours.
        
        Returns:
            True if allowed to trade
        """
        now = datetime.now().time()
        
        # Market open is 9:30 AM
        market_open = time(9, 30)
        skip_until = time(9, 30 + self.skip_first_minutes)
        
        # Market close is 4:00 PM
        market_close = time(16, 0)
        reduce_after = time(16 - self.reduce_last_minutes // 60,
                           (16 * 60 - self.reduce_last_minutes) % 60)
        
        # Check if we're in the skip period
        if market_open <= now < skip_until:
            return False
        
        # Check if we're near close
        if now >= reduce_after:
            return False
        
        return True
    
    def on_bar(self, bar: Bar) -> List[Signal]:
        """
        Process new bar and generate signals.
        
        Args:
            bar: New bar data
            
        Returns:
            List of signals
        """
        if not self.enabled:
            return []
        
        symbol = bar.symbol
        
        # Check if we're in allowed trading time
        if not self._is_trading_time():
            return []
        
        # Get or create state for this symbol
        if symbol not in self.state:
            # Initialize with minimal data
            self.state[symbol] = pd.DataFrame()
        
        df = self.state[symbol]
        
        # Add new bar to state
        new_row = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'vwap': bar.vwap if bar.vwap else None
        }
        
        new_df = pd.DataFrame([new_row], index=[bar.ts])
        df = pd.concat([df, new_df])
        
        # Need minimum bars for indicators
        if len(df) < max(self.zscore_window, self.atr_window):
            self._update_state(symbol, df)
            return []
        
        # Calculate indicators
        try:
            df = self._calculate_indicators(df)
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators for {symbol}: {e}")
            return []
        
        # Update state
        self._update_state(symbol, df)
        
        # Get latest values
        latest = df.iloc[-1]
        z = latest['zscore_vwap']
        atr_val = latest['atr']
        volume = latest['volume']
        volume_ma = latest['volume_ma']
        rsi_val = latest['rsi']
        
        # Check for NaN values
        if pd.isna(z) or pd.isna(atr_val) or atr_val <= 0:
            return []
        
        # Volume filter
        if volume < self.min_volume or volume < volume_ma * 0.5:
            return []
        
        # Generate signals
        signals = []
        
        # Long signal (oversold)
        if z <= -self.zscore_entry:
            # Additional confirmation: RSI also oversold
            if not pd.isna(rsi_val) and rsi_val < 35:
                confidence = min(1.0, abs(z) / (self.zscore_entry + 1.0))
                signal = Signal(
                    symbol=symbol,
                    strategy_name=self.name,
                    strength=1.0,  # Full buy
                    confidence=confidence,
                    reason=f"Oversold: zscore={z:.2f}, RSI={rsi_val:.1f}",
                    metadata={
                        'zscore': float(z),
                        'atr': float(atr_val),
                        'rsi': float(rsi_val),
                        'entry_price': float(bar.close)
                    }
                )
                signals.append(signal)
                self._record_signal(signal)
        
        # Short signal (overbought)
        elif z >= self.zscore_entry:
            # Additional confirmation: RSI also overbought
            if not pd.isna(rsi_val) and rsi_val > 65:
                confidence = min(1.0, abs(z) / (self.zscore_entry + 1.0))
                signal = Signal(
                    symbol=symbol,
                    strategy_name=self.name,
                    strength=-1.0,  # Full sell/short
                    confidence=confidence,
                    reason=f"Overbought: zscore={z:.2f}, RSI={rsi_val:.1f}",
                    metadata={
                        'zscore': float(z),
                        'atr': float(atr_val),
                        'rsi': float(rsi_val),
                        'entry_price': float(bar.close)
                    }
                )
                signals.append(signal)
                self._record_signal(signal)
        
        # Exit signal (mean reversion completed)
        elif abs(z) <= self.zscore_exit:
            # Check if we had a previous signal that needs exit
            last_signal = self.get_last_signal(symbol)
            if last_signal and abs(last_signal.strength) > 0.5:
                signal = Signal(
                    symbol=symbol,
                    strategy_name=self.name,
                    strength=0.0,  # Neutral/exit
                    confidence=0.8,
                    reason=f"Mean reversion: zscore={z:.2f}",
                    metadata={
                        'zscore': float(z),
                        'exit_price': float(bar.close)
                    }
                )
                signals.append(signal)
                self._record_signal(signal)
        
        return signals
    
    def on_end_of_day(self) -> None:
        """
        End of day processing - flatten all positions.
        """
        if self.flatten_eod:
            logger.info(f"{self.name}: End of day - generating exit signals")
            
            # Generate exit signals for all tracked symbols
            for symbol in list(self.state.keys()):
                last_signal = self.get_last_signal(symbol)
                if last_signal and abs(last_signal.strength) > 0:
                    exit_signal = Signal(
                        symbol=symbol,
                        strategy_name=self.name,
                        strength=0.0,
                        confidence=1.0,
                        reason="End of day flatten",
                        metadata={'eod_flatten': True}
                    )
                    self._record_signal(exit_signal)
        
        # Could also clear state here if desired
        # self.state.clear()
