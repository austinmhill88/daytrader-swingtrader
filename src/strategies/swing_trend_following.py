"""
Swing trend following strategy using EMA crossovers and ADX.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

from src.strategies.base import Strategy
from src.models import Signal, Bar
from src.utils import ema, atr, adx
from loguru import logger


class SwingTrendFollowing(Strategy):
    """
    Swing trend following strategy using EMA crossovers with ADX filter.
    
    Entry conditions:
    - Fast EMA crosses above slow EMA (uptrend)
    - Fast EMA crosses below slow EMA (downtrend)
    - ADX above threshold (trending market)
    - Pullback to fast EMA within trend
    
    Exit conditions:
    - Trend reversal (EMA crossover opposite direction)
    - ATR-based trailing stop
    - Maximum holding period exceeded
    """
    
    name = "swing_trend_following"
    
    def __init__(self, config: Dict, full_config: Dict = None):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration
            full_config: Full application configuration (optional, for ML model access)
        """
        super().__init__(config)
        
        self.full_config = full_config or {}
        
        # Strategy parameters
        self.ema_fast = config.get('ema_fast', 20)
        self.ema_slow = config.get('ema_slow', 50)
        self.adx_threshold = config.get('adx_threshold', 20)
        self.min_adx = config.get('min_adx', 15)
        self.atr_stop_multiplier = config.get('atr_stop_multiplier', 2.0)
        self.trailing_atr_multiplier = config.get('trailing_atr_multiplier', 3.0)
        self.max_positions = config.get('max_positions', 10)
        self.holding_period_min = config.get('holding_period_min', 1)
        self.holding_period_max = config.get('holding_period_max', 10)
        self.pullback_threshold_pct = config.get('pullback_threshold_pct', 2.0)
        
        # Position entry times tracking
        self.entry_times: Dict[str, datetime] = {}
        
        # ML model support
        self.ml_enabled = config.get('ml_model', {}).get('enabled', False)
        self.model = None
        self.registry = None
        self.fs = None
        
        logger.info(
            f"{self.name} initialized | "
            f"EMA({self.ema_fast}/{self.ema_slow}), "
            f"ADX>{self.adx_threshold}, "
            f"ml_enabled={self.ml_enabled}"
        )
    
    def warmup(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Warm up with historical data.
        
        Args:
            historical_data: Dictionary mapping symbol to DataFrame
        """
        logger.info(f"{self.name}: Warming up with data for {len(historical_data)} symbols")
        
        # Load ML model if enabled
        if self.ml_enabled and self.full_config:
            try:
                from pathlib import Path
                import joblib
                from src.model_registry import ModelRegistry
                from src.feature_store import FeatureStore
                
                # Initialize registry and feature store
                storage_cfg = self.full_config.get('storage', {})
                model_dir = storage_cfg.get('model_dir', './data/models')
                feature_dir = storage_cfg.get('feature_store_dir', './data/features')
                
                self.registry = ModelRegistry(model_dir)
                self.fs = FeatureStore(base_dir=feature_dir)
                
                # Load latest model
                artifact = self.registry.get_latest_model("swing_trend_following")
                if artifact and Path(artifact).exists():
                    self.model = joblib.load(artifact)
                    metrics = self.registry.get_metrics("swing_trend_following")
                    logger.info(
                        f"{self.name}: Loaded ML model | "
                        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                        f"Win rate: {metrics.get('win_rate', 0):.2%}"
                    )
                else:
                    logger.info(f"{self.name}: No promoted ML model found, using rule-based only")
            except Exception as e:
                logger.warning(f"{self.name}: Error loading ML model: {e}")
                self.ml_enabled = False
        
        for symbol, df in historical_data.items():
            if df is None or len(df) == 0:
                continue
            
            try:
                # Calculate indicators
                df = self._calculate_indicators(df)
                
                # Store limited history
                self._update_state(symbol, df.iloc[-500:].copy())
                
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
        # EMAs
        df['ema_fast'] = ema(df['close'], self.ema_fast)
        df['ema_slow'] = ema(df['close'], self.ema_slow)
        
        # Trend direction
        df['trend'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'trend'] = 1  # Uptrend
        df.loc[df['ema_fast'] < df['ema_slow'], 'trend'] = -1  # Downtrend
        
        # Trend strength (ADX)
        try:
            df['adx'] = adx(df['high'], df['low'], df['close'], period=14)
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            df['adx'] = 0
        
        # ATR for stops
        df['atr'] = atr(df['high'], df['low'], df['close'], window=20)
        
        # Distance from EMAs (for pullback detection)
        df['dist_from_fast_ema'] = (df['close'] - df['ema_fast']) / df['ema_fast'] * 100
        df['dist_from_slow_ema'] = (df['close'] - df['ema_slow']) / df['ema_slow'] * 100
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df
    
    def _detect_crossover(self, df: pd.DataFrame) -> int:
        """
        Detect EMA crossover.
        
        Args:
            df: DataFrame with trend column
            
        Returns:
            1 for bullish crossover, -1 for bearish crossover, 0 for no crossover
        """
        if len(df) < 2:
            return 0
        
        current_trend = df['trend'].iloc[-1]
        previous_trend = df['trend'].iloc[-2]
        
        # Bullish crossover (from downtrend/neutral to uptrend)
        if current_trend == 1 and previous_trend != 1:
            return 1
        
        # Bearish crossover (from uptrend/neutral to downtrend)
        elif current_trend == -1 and previous_trend != -1:
            return -1
        
        return 0
    
    def _is_pullback_entry(self, df: pd.DataFrame, trend: int) -> bool:
        """
        Check if price has pulled back to fast EMA (good entry in existing trend).
        
        Args:
            df: DataFrame with indicators
            trend: Current trend direction (1 or -1)
            
        Returns:
            True if pullback entry conditions met
        """
        if len(df) < 3:
            return False
        
        latest = df.iloc[-1]
        dist = latest['dist_from_fast_ema']
        
        # In uptrend, look for price near or slightly below fast EMA
        if trend == 1:
            if -self.pullback_threshold_pct <= dist <= 1.0:
                # Check that we actually pulled back (previous bar was further away)
                prev_dist = df.iloc[-2]['dist_from_fast_ema']
                if prev_dist < dist:  # Price moving back toward EMA
                    return True
        
        # In downtrend, look for price near or slightly above fast EMA
        elif trend == -1:
            if -1.0 <= dist <= self.pullback_threshold_pct:
                # Check that we actually pulled back (previous bar was further away)
                prev_dist = df.iloc[-2]['dist_from_fast_ema']
                if prev_dist > dist:  # Price moving back toward EMA
                    return True
        
        return False
    
    def _should_exit_position(self, symbol: str, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Check if position should be exited.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with indicators
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if symbol not in self.entry_times:
            return False, ""
        
        latest = df.iloc[-1]
        current_trend = latest['trend']
        
        # Check holding period
        entry_time = self.entry_times[symbol]
        holding_days = (datetime.now() - entry_time).days
        
        if holding_days > self.holding_period_max:
            return True, f"Max holding period ({self.holding_period_max} days) exceeded"
        
        # Check trend reversal
        last_signal = self.get_last_signal(symbol)
        if last_signal:
            # Exit long if trend turns bearish
            if last_signal.strength > 0 and current_trend == -1:
                return True, "Trend reversed (bearish)"
            
            # Exit short if trend turns bullish
            if last_signal.strength < 0 and current_trend == 1:
                return True, "Trend reversed (bullish)"
        
        # Check ADX - if trend weakens significantly
        if not pd.isna(latest['adx']) and latest['adx'] < self.min_adx:
            return True, f"Trend weakened (ADX={latest['adx']:.1f})"
        
        return False, ""
    
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
        
        # Get or create state for this symbol
        if symbol not in self.state:
            self.state[symbol] = pd.DataFrame()
        
        df = self.state[symbol]
        
        # Add new bar to state
        new_row = {
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        
        new_df = pd.DataFrame([new_row], index=[bar.ts])
        df = pd.concat([df, new_df])
        
        # Need minimum bars for EMAs
        if len(df) < max(self.ema_slow, 50):
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
        current_trend = latest['trend']
        adx_val = latest['adx']
        atr_val = latest['atr']
        volume = latest['volume']
        volume_ma = latest['volume_ma']
        
        # Check for NaN values
        if pd.isna(atr_val) or atr_val <= 0:
            return []
        
        signals = []
        
        # Check for exit conditions first
        should_exit, exit_reason = self._should_exit_position(symbol, df)
        if should_exit:
            signal = Signal(
                symbol=symbol,
                strategy_name=self.name,
                strength=0.0,  # Exit
                confidence=0.9,
                reason=exit_reason,
                metadata={
                    'exit_price': float(bar.close),
                    'trend': int(current_trend),
                    'adx': float(adx_val) if not pd.isna(adx_val) else None
                }
            )
            signals.append(signal)
            self._record_signal(signal)
            
            # Remove from tracking
            if symbol in self.entry_times:
                del self.entry_times[symbol]
            
            return signals
        
        # Check ADX threshold (need trending market)
        if pd.isna(adx_val) or adx_val < self.adx_threshold:
            return []
        
        # Volume filter
        if not pd.isna(volume_ma) and volume < volume_ma * 0.7:
            return []
        
        # Detect crossover
        crossover = self._detect_crossover(df)
        
        # Check for pullback entry
        is_pullback = self._is_pullback_entry(df, current_trend)
        
        # Generate entry signals
        
        # Bullish signal (uptrend)
        if (crossover == 1 or (current_trend == 1 and is_pullback)):
            entry_type = "Crossover" if crossover == 1 else "Pullback"
            confidence = 0.8 if crossover == 1 else 0.6
            
            signal = Signal(
                symbol=symbol,
                strategy_name=self.name,
                strength=0.7,  # Moderate buy (swing trading)
                confidence=confidence,
                reason=f"{entry_type} entry: Uptrend, ADX={adx_val:.1f}",
                metadata={
                    'trend': 1,
                    'adx': float(adx_val),
                    'atr': float(atr_val),
                    'entry_price': float(bar.close),
                    'entry_type': entry_type
                }
            )
            signals.append(signal)
            self._record_signal(signal)
            self.entry_times[symbol] = datetime.now()
        
        # Bearish signal (downtrend)
        elif (crossover == -1 or (current_trend == -1 and is_pullback)):
            entry_type = "Crossover" if crossover == -1 else "Pullback"
            confidence = 0.8 if crossover == -1 else 0.6
            
            signal = Signal(
                symbol=symbol,
                strategy_name=self.name,
                strength=-0.7,  # Moderate sell/short (swing trading)
                confidence=confidence,
                reason=f"{entry_type} entry: Downtrend, ADX={adx_val:.1f}",
                metadata={
                    'trend': -1,
                    'adx': float(adx_val),
                    'atr': float(atr_val),
                    'entry_price': float(bar.close),
                    'entry_type': entry_type
                }
            )
            signals.append(signal)
            self._record_signal(signal)
            self.entry_times[symbol] = datetime.now()
        
        return signals
    
    def on_end_of_day(self) -> None:
        """
        End of day processing.
        """
        # Swing strategy holds overnight, so no automatic flatten
        # But we can log summary
        active_positions = len(self.entry_times)
        if active_positions > 0:
            logger.info(
                f"{self.name}: End of day | "
                f"Active swing positions: {active_positions}"
            )
