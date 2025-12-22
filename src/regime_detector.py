"""
Regime detection for market conditions.
Identifies volatility regimes and market states to adjust strategy behavior.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum


class RegimeType(Enum):
    """Market regime types."""
    LOW_VOL = "low_volatility"
    MEDIUM_VOL = "medium_volatility"
    HIGH_VOL = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    CRISIS = "crisis"


class RegimeDetector:
    """
    Detects market regimes based on volatility, breadth, and correlation.
    Phase 1 implementation - uses threshold-based detection.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize regime detector.
        
        Args:
            config: Regime configuration from config.yaml
        """
        self.config = config
        regime_config = config.get('regime', {})
        
        self.enabled = regime_config.get('enabled', False)
        self.method = regime_config.get('method', 'threshold')
        
        # Volatility thresholds
        vol_regimes = regime_config.get('volatility_regimes', {})
        self.low_vol_threshold = vol_regimes.get('low', 0.10)
        self.medium_vol_threshold = vol_regimes.get('medium', 0.20)
        self.high_vol_threshold = vol_regimes.get('high', 0.30)
        
        # Current regime state
        self.current_regime: Optional[RegimeType] = None
        self.regime_start_time: Optional[datetime] = None
        self.regime_history: List[Dict] = []
        
        # Market data cache for regime calculation
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        logger.info(
            f"RegimeDetector initialized | "
            f"Enabled: {self.enabled}, Method: {self.method}"
        )
    
    def update_market_data(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Update market data for regime calculation.
        
        Args:
            symbol: Symbol (e.g., 'SPY', 'QQQ')
            df: DataFrame with OHLCV data
        """
        self.market_data[symbol] = df.tail(100).copy()  # Keep last 100 bars
    
    def calculate_regime_features(self, reference_symbol: str = 'SPY') -> Dict[str, float]:
        """
        Calculate regime features.
        
        Args:
            reference_symbol: Reference symbol for regime calculation (default: SPY)
            
        Returns:
            Dictionary with regime features
        """
        if reference_symbol not in self.market_data:
            logger.warning(f"No data for {reference_symbol}, cannot calculate regime")
            return {}
        
        df = self.market_data[reference_symbol]
        
        if len(df) < 20:
            logger.warning(f"Insufficient data for regime calculation")
            return {}
        
        features = {}
        
        # 1. Realized Volatility (annualized)
        df['returns'] = df['close'].pct_change()
        realized_vol = df['returns'].std() * np.sqrt(252)
        features['realized_vol'] = realized_vol
        
        # 2. Trend Strength (using ADX-like calculation)
        # Simplified: use EMA slope
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema_slope'] = (df['ema20'] - df['ema20'].shift(5)) / df['ema20'].shift(5)
        trend_strength = abs(df['ema_slope'].iloc[-1])
        features['trend_strength'] = trend_strength
        
        # 3. Trend Direction
        ema_slope = df['ema_slope'].iloc[-1]
        if ema_slope > 0.01:
            features['trend_direction'] = 1  # Up
        elif ema_slope < -0.01:
            features['trend_direction'] = -1  # Down
        else:
            features['trend_direction'] = 0  # Neutral/ranging
        
        # 4. VIX Proxy (using high-low range)
        df['range_pct'] = (df['high'] - df['low']) / df['close']
        vix_proxy = df['range_pct'].rolling(20).mean().iloc[-1]
        features['vix_proxy'] = vix_proxy
        
        # 5. Market Breadth (if we have multiple symbols)
        if len(self.market_data) > 1:
            positive_symbols = 0
            total_symbols = 0
            
            for symbol, symbol_df in self.market_data.items():
                if len(symbol_df) >= 20:
                    # Use a copy to avoid modifying cached data
                    df_copy = symbol_df.copy()
                    df_copy['returns'] = df_copy['close'].pct_change()
                    if df_copy['returns'].iloc[-1] > 0:
                        positive_symbols += 1
                    total_symbols += 1
            
            if total_symbols > 0:
                features['breadth'] = positive_symbols / total_symbols
            else:
                features['breadth'] = 0.5  # Neutral
        else:
            features['breadth'] = 0.5  # Neutral (no data)
        
        # 6. Correlation (if we have SPY and QQQ)
        if 'SPY' in self.market_data and 'QQQ' in self.market_data:
            spy_df = self.market_data['SPY']
            qqq_df = self.market_data['QQQ']
            
            if len(spy_df) >= 20 and len(qqq_df) >= 20:
                spy_returns = spy_df['close'].pct_change().tail(20)
                qqq_returns = qqq_df['close'].pct_change().tail(20)
                correlation = spy_returns.corr(qqq_returns)
                features['spy_qqq_correlation'] = correlation
        
        return features
    
    def detect_regime(self, reference_symbol: str = 'SPY') -> RegimeType:
        """
        Detect current market regime.
        
        Args:
            reference_symbol: Reference symbol for regime detection
            
        Returns:
            Current regime type
        """
        if not self.enabled:
            return RegimeType.MEDIUM_VOL  # Default
        
        features = self.calculate_regime_features(reference_symbol)
        
        if not features:
            return RegimeType.MEDIUM_VOL  # Default
        
        # Threshold-based regime detection
        realized_vol = features.get('realized_vol', 0.15)
        trend_strength = features.get('trend_strength', 0)
        trend_direction = features.get('trend_direction', 0)
        vix_proxy = features.get('vix_proxy', 0.02)
        
        # Detect volatility regime
        if realized_vol < self.low_vol_threshold:
            vol_regime = RegimeType.LOW_VOL
        elif realized_vol < self.medium_vol_threshold:
            vol_regime = RegimeType.MEDIUM_VOL
        elif realized_vol < self.high_vol_threshold:
            vol_regime = RegimeType.HIGH_VOL
        else:
            vol_regime = RegimeType.CRISIS
        
        # Detect trend regime
        if trend_strength > 0.02 and trend_direction > 0:
            trend_regime = RegimeType.TRENDING_UP
        elif trend_strength > 0.02 and trend_direction < 0:
            trend_regime = RegimeType.TRENDING_DOWN
        else:
            trend_regime = RegimeType.RANGING
        
        # Combined regime (prioritize crisis/high vol)
        if vol_regime == RegimeType.CRISIS:
            regime = RegimeType.CRISIS
        elif vol_regime == RegimeType.HIGH_VOL:
            regime = RegimeType.HIGH_VOL
        else:
            # Use trend regime for low/medium vol
            regime = trend_regime
        
        # Update regime if changed
        if regime != self.current_regime:
            self._update_regime(regime, features)
        
        return regime
    
    def _update_regime(self, new_regime: RegimeType, features: Dict) -> None:
        """
        Update current regime and log transition.
        
        Args:
            new_regime: New regime type
            features: Regime features at time of transition
        """
        old_regime = self.current_regime
        self.current_regime = new_regime
        
        now = datetime.now()
        
        # Record transition
        if old_regime is not None:
            duration = (now - self.regime_start_time).total_seconds() / 60  # minutes
            
            logger.info(
                f"Regime transition | "
                f"{old_regime.value} -> {new_regime.value} | "
                f"Duration: {duration:.1f}m | "
                f"RealizedVol: {features.get('realized_vol', 0):.3f}"
            )
            
            self.regime_history.append({
                'start_time': self.regime_start_time,
                'end_time': now,
                'regime': old_regime.value,
                'duration_minutes': duration
            })
        else:
            logger.info(f"Initial regime detected: {new_regime.value}")
        
        self.regime_start_time = now
    
    def get_position_size_multiplier(self, strategy_type: str = "intraday") -> float:
        """
        Get position size multiplier based on regime.
        
        Args:
            strategy_type: "intraday" or "swing"
            
        Returns:
            Multiplier for position sizing (0.0 to 1.0)
        """
        if not self.enabled or self.current_regime is None:
            return 1.0  # No adjustment
        
        regime = self.current_regime
        
        # Adjust sizing based on regime
        if regime == RegimeType.CRISIS:
            # Significantly reduce size in crisis
            return 0.25
        elif regime == RegimeType.HIGH_VOL:
            # Reduce size in high vol
            return 0.5
        elif regime == RegimeType.LOW_VOL:
            # Can use larger size in low vol
            return 1.0
        elif regime == RegimeType.MEDIUM_VOL:
            # Standard size
            return 1.0
        elif regime == RegimeType.TRENDING_UP:
            # Favor swing strategies in uptrend
            return 1.2 if strategy_type == "swing" else 0.9
        elif regime == RegimeType.TRENDING_DOWN:
            # Reduce long exposure in downtrend
            return 0.8
        elif regime == RegimeType.RANGING:
            # Favor mean reversion in ranging market
            return 1.2 if strategy_type == "intraday" else 0.8
        
        return 1.0
    
    def should_enable_strategy(self, strategy_name: str) -> bool:
        """
        Determine if strategy should be active in current regime.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            True if strategy should be enabled
        """
        if not self.enabled or self.current_regime is None:
            return True  # No adjustment
        
        regime = self.current_regime
        
        # Disable all strategies in crisis
        if regime == RegimeType.CRISIS:
            return False
        
        # Strategy-specific regime preferences
        if "mean_reversion" in strategy_name.lower():
            # Mean reversion works better in ranging/low vol
            return regime in [
                RegimeType.RANGING,
                RegimeType.LOW_VOL,
                RegimeType.MEDIUM_VOL
            ]
        
        if "trend" in strategy_name.lower():
            # Trend following works better in trending markets
            return regime in [
                RegimeType.TRENDING_UP,
                RegimeType.TRENDING_DOWN,
                RegimeType.MEDIUM_VOL
            ]
        
        # Default: enable in all non-crisis regimes
        return True
    
    def get_regime_summary(self) -> Dict:
        """
        Get summary of current regime.
        
        Returns:
            Dictionary with regime information
        """
        if not self.enabled:
            return {
                'enabled': False,
                'current_regime': 'N/A'
            }
        
        duration_minutes = 0
        if self.regime_start_time:
            duration_minutes = (datetime.now() - self.regime_start_time).total_seconds() / 60
        
        return {
            'enabled': True,
            'current_regime': self.current_regime.value if self.current_regime else 'Unknown',
            'duration_minutes': duration_minutes,
            'num_transitions': len(self.regime_history)
        }
