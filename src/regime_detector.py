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
        Calculate regime features with multi-signal approach.
        Includes trend, volatility, breadth, and liquidity indicators.
        
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
        
        # 2. Trend Strength (using EMA slope)
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
        
        # 5. Market Breadth (advancers vs decliners)
        if len(self.market_data) > 1:
            positive_symbols = 0
            total_symbols = 0
            advancing_symbols = []
            declining_symbols = []
            
            for symbol, symbol_df in self.market_data.items():
                if len(symbol_df) >= 20:
                    # Use a copy to avoid modifying cached data
                    df_copy = symbol_df.copy()
                    df_copy['returns'] = df_copy['close'].pct_change()
                    
                    # Check today's return
                    if df_copy['returns'].iloc[-1] > 0:
                        positive_symbols += 1
                        advancing_symbols.append(symbol)
                    else:
                        declining_symbols.append(symbol)
                    total_symbols += 1
            
            if total_symbols > 0:
                breadth = positive_symbols / total_symbols
                features['breadth'] = breadth
                features['advance_decline_ratio'] = positive_symbols / max(1, total_symbols - positive_symbols)
                features['num_advancing'] = positive_symbols
                features['num_declining'] = total_symbols - positive_symbols
            else:
                features['breadth'] = 0.5  # Neutral
                features['advance_decline_ratio'] = 1.0
        else:
            features['breadth'] = 0.5  # Neutral (no data)
            features['advance_decline_ratio'] = 1.0
        
        # 6. Percent Above Moving Average (breadth indicator)
        if len(self.market_data) > 5:
            above_ma = 0
            total = 0
            
            for symbol, symbol_df in self.market_data.items():
                if len(symbol_df) >= 50:
                    df_copy = symbol_df.copy()
                    df_copy['ma50'] = df_copy['close'].rolling(50).mean()
                    
                    if df_copy['close'].iloc[-1] > df_copy['ma50'].iloc[-1]:
                        above_ma += 1
                    total += 1
            
            if total > 0:
                features['pct_above_ma50'] = above_ma / total
            else:
                features['pct_above_ma50'] = 0.5
        else:
            features['pct_above_ma50'] = 0.5
        
        # 7. Correlation (if we have SPY and QQQ)
        if 'SPY' in self.market_data and 'QQQ' in self.market_data:
            spy_df = self.market_data['SPY']
            qqq_df = self.market_data['QQQ']
            
            if len(spy_df) >= 20 and len(qqq_df) >= 20:
                spy_returns = spy_df['close'].pct_change().tail(20)
                qqq_returns = qqq_df['close'].pct_change().tail(20)
                correlation = spy_returns.corr(qqq_returns)
                features['spy_qqq_correlation'] = correlation
        
        # 8. Liquidity indicator (average spread if available)
        # This would require bid/ask data - placeholder for now
        features['liquidity_score'] = 1.0  # Placeholder (1.0 = normal liquidity)
        
        # 9. Returns slope (linear regression on recent returns)
        if len(df) >= 20:
            recent_returns = df['returns'].tail(20).values
            x = np.arange(len(recent_returns))
            if len(recent_returns) > 1 and not np.all(np.isnan(recent_returns)):
                # Simple linear regression slope
                slope = np.polyfit(x[~np.isnan(recent_returns)], 
                                 recent_returns[~np.isnan(recent_returns)], 1)[0]
                features['returns_slope'] = slope
            else:
                features['returns_slope'] = 0.0
        
        return features
    
    def detect_regime(self, reference_symbol: str = 'SPY') -> RegimeType:
        """
        Detect current market regime using multi-signal approach.
        Considers trend, volatility, breadth, and liquidity.
        
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
        
        # Extract features
        realized_vol = features.get('realized_vol', 0.15)
        trend_strength = features.get('trend_strength', 0)
        trend_direction = features.get('trend_direction', 0)
        vix_proxy = features.get('vix_proxy', 0.02)
        breadth = features.get('breadth', 0.5)
        pct_above_ma50 = features.get('pct_above_ma50', 0.5)
        
        # Detect volatility regime
        if realized_vol < self.low_vol_threshold:
            vol_regime = RegimeType.LOW_VOL
        elif realized_vol < self.medium_vol_threshold:
            vol_regime = RegimeType.MEDIUM_VOL
        elif realized_vol < self.high_vol_threshold:
            vol_regime = RegimeType.HIGH_VOL
        else:
            vol_regime = RegimeType.CRISIS
        
        # Detect trend regime with breadth confirmation
        # Strong trend requires both directional momentum AND breadth support
        if trend_strength > 0.02:
            if trend_direction > 0:
                # Uptrend - check breadth confirmation
                if breadth > 0.6 or pct_above_ma50 > 0.6:
                    trend_regime = RegimeType.TRENDING_UP
                else:
                    # Weak breadth suggests false breakout
                    trend_regime = RegimeType.RANGING
            elif trend_direction < 0:
                # Downtrend - check breadth confirmation
                if breadth < 0.4 or pct_above_ma50 < 0.4:
                    trend_regime = RegimeType.TRENDING_DOWN
                else:
                    # Weak breadth suggests false breakdown
                    trend_regime = RegimeType.RANGING
            else:
                trend_regime = RegimeType.RANGING
        else:
            trend_regime = RegimeType.RANGING
        
        # Combined regime (prioritize crisis/high vol, then incorporate breadth)
        if vol_regime == RegimeType.CRISIS:
            regime = RegimeType.CRISIS
        elif vol_regime == RegimeType.HIGH_VOL:
            # High vol can still have trend
            if trend_regime in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]:
                # High vol + trend = still use trend regime but risk is elevated
                regime = trend_regime
            else:
                regime = RegimeType.HIGH_VOL
        else:
            # Low/medium vol - use trend regime
            regime = trend_regime
        
        # Update regime if changed
        if regime != self.current_regime:
            self._update_regime(regime, features)
        
        return regime
    
    def _update_regime(self, new_regime: RegimeType, features: Dict) -> None:
        """
        Update current regime and log transition with detailed metrics.
        
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
            
            # Enhanced logging with key metrics
            logger.info(
                f"🔄 Regime Transition | "
                f"{old_regime.value} → {new_regime.value} | "
                f"Duration: {duration:.1f}m | "
                f"Vol: {features.get('realized_vol', 0):.2%}, "
                f"Breadth: {features.get('breadth', 0.5):.1%}, "
                f"Trend: {features.get('trend_strength', 0):.3f}, "
                f"%>MA50: {features.get('pct_above_ma50', 0.5):.1%}"
            )
            
            # Record detailed transition
            transition_record = {
                'start_time': self.regime_start_time.isoformat(),
                'end_time': now.isoformat(),
                'old_regime': old_regime.value,
                'new_regime': new_regime.value,
                'duration_minutes': duration,
                'features_at_transition': {
                    'realized_vol': features.get('realized_vol', 0),
                    'breadth': features.get('breadth', 0.5),
                    'trend_strength': features.get('trend_strength', 0),
                    'trend_direction': features.get('trend_direction', 0),
                    'pct_above_ma50': features.get('pct_above_ma50', 0.5),
                    'advance_decline_ratio': features.get('advance_decline_ratio', 1.0),
                    'vix_proxy': features.get('vix_proxy', 0),
                    'returns_slope': features.get('returns_slope', 0)
                }
            }
            
            self.regime_history.append(transition_record)
            
            # Keep last 100 transitions
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
        else:
            logger.info(
                f"📊 Initial Regime | {new_regime.value} | "
                f"Vol: {features.get('realized_vol', 0):.2%}, "
                f"Breadth: {features.get('breadth', 0.5):.1%}"
            )
        
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
        Get summary of current regime with detailed metrics.
        
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
        
        # Get current features
        current_features = self.calculate_regime_features()
        
        return {
            'enabled': True,
            'current_regime': self.current_regime.value if self.current_regime else 'Unknown',
            'duration_minutes': duration_minutes,
            'num_transitions': len(self.regime_history),
            'features': {
                'realized_vol': current_features.get('realized_vol', 0),
                'breadth': current_features.get('breadth', 0.5),
                'pct_above_ma50': current_features.get('pct_above_ma50', 0.5),
                'trend_strength': current_features.get('trend_strength', 0),
                'advance_decline_ratio': current_features.get('advance_decline_ratio', 1.0)
            }
        }
    
    def get_regime_history_stats(self) -> Dict:
        """
        Get statistics on regime history.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {}
        
        # Count time in each regime
        regime_durations = {}
        for record in self.regime_history:
            regime = record.get('old_regime', record.get('regime'))
            duration = record.get('duration_minutes', 0)
            
            if regime in regime_durations:
                regime_durations[regime] += duration
            else:
                regime_durations[regime] = duration
        
        # Calculate percentages
        total_duration = sum(regime_durations.values())
        regime_percentages = {
            regime: (duration / total_duration * 100) if total_duration > 0 else 0
            for regime, duration in regime_durations.items()
        }
        
        # Average features per regime
        regime_features = {}
        for record in self.regime_history:
            if 'features_at_transition' in record:
                regime = record['old_regime']
                if regime not in regime_features:
                    regime_features[regime] = []
                regime_features[regime].append(record['features_at_transition'])
        
        # Average features
        avg_features_by_regime = {}
        for regime, feature_list in regime_features.items():
            if feature_list:
                avg_features = {}
                for key in feature_list[0].keys():
                    values = [f.get(key, 0) for f in feature_list]
                    avg_features[key] = np.mean(values)
                avg_features_by_regime[regime] = avg_features
        
        return {
            'regime_durations_minutes': regime_durations,
            'regime_percentages': regime_percentages,
            'total_transitions': len(self.regime_history),
            'avg_features_by_regime': avg_features_by_regime
        }
