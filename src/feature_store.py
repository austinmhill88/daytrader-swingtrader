"""
Feature store for ML model features (Phase 2).
Computes and stores deterministic features with versioning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json
from loguru import logger

from src.utils import (
    zscore, atr, ema, rsi, adx, bollinger_bands,
    macd, stochastic, calculate_returns, calculate_volatility
)


class FeatureStore:
    """
    Feature computation and storage with versioning.
    Phase 2 implementation for ML model training.
    """
    
    def __init__(self, config: Dict[str, Any] = None, base_dir: str = None):
        """
        Initialize feature store.
        
        Args:
            config: Application configuration (optional if base_dir provided)
            base_dir: Base directory for features (optional if config provided)
        """
        if config:
            self.config = config
            self.feature_dir = Path(config.get('storage', {}).get('feature_store_dir', './data/features'))
        elif base_dir:
            self.config = {}
            self.feature_dir = Path(base_dir)
        else:
            raise ValueError("Either config or base_dir must be provided")
        
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature definitions (extensible)
        self.feature_definitions = {
            'price_features': ['returns', 'log_returns', 'volatility'],
            'technical_features': ['rsi', 'atr', 'ema_fast', 'ema_slow', 'adx'],
            'volume_features': ['volume_zscore', 'volume_ma_ratio'],
            'microstructure_features': ['spread', 'vwap_distance'],
            'regime_features': ['realized_vol', 'trend_strength']
        }
        
        logger.info(f"FeatureStore initialized | Path: {self.feature_dir}")
    
    def compute_features(
        self,
        df: pd.DataFrame,
        feature_set: str = 'all'
    ) -> pd.DataFrame:
        """
        Compute features for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            feature_set: Which features to compute ('all', 'intraday', 'swing')
            
        Returns:
            DataFrame with added feature columns
        """
        try:
            df = df.copy()
            
            # Price features
            df['returns'] = calculate_returns(df['close'])
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = calculate_volatility(df['close'], window=20)
            
            # Technical indicators
            df['rsi'] = rsi(df['close'], period=14)
            df['atr'] = atr(df['high'], df['low'], df['close'], window=14)
            df['ema_fast'] = ema(df['close'], span=20)
            df['ema_slow'] = ema(df['close'], span=50)
            df['adx'] = adx(df['high'], df['low'], df['close'], period=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            macd_line, signal_line, histogram = macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = histogram
            
            # Stochastic
            stoch_k, stoch_d = stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_zscore'] = zscore(df['volume'], window=20)
            df['volume_ma_ratio'] = df['volume'] / df['volume_ma']
            
            # Microstructure features
            if 'vwap' in df.columns:
                df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
            
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
            df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
            
            # Price momentum features
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            df['momentum_60'] = df['close'] / df['close'].shift(60) - 1
            
            # Trend strength
            df['ema_slope'] = df['ema_fast'].diff(5) / df['ema_fast'].shift(5)
            df['trend_strength'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']
            
            # Regime features
            df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
            # Use inf for upper bound to handle extreme volatility
            df['vol_regime'] = pd.cut(
                df['realized_vol'],
                bins=[0, 0.15, 0.25, float('inf')],
                labels=['low', 'med', 'high']
            )
            
            logger.debug(f"Computed {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return df
    
    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Save features to disk with versioning.
        
        Args:
            df: DataFrame with computed features
            symbol: Stock symbol
            timeframe: Data timeframe
            version: Optional version string
            
        Returns:
            True if successful
        """
        try:
            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create symbol directory
            symbol_dir = self.feature_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features as parquet
            filename = f"{timeframe}_{version}.parquet"
            filepath = symbol_dir / filename
            df.to_parquet(filepath, compression='snappy')
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'version': version,
                'num_rows': len(df),
                'num_features': len(df.columns),
                'features': list(df.columns),
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                },
                'hash': self._compute_hash(df)
            }
            
            metadata_file = symbol_dir / f"{timeframe}_{version}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved features for {symbol} ({len(df)} rows) to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return False
    
    def load_features(
        self,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load features from disk.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            version: Optional specific version (latest if None)
            
        Returns:
            DataFrame with features or None
        """
        try:
            symbol_dir = self.feature_dir / symbol
            if not symbol_dir.exists():
                logger.warning(f"No features found for {symbol}")
                return None
            
            # Find matching files
            if version:
                pattern = f"{timeframe}_{version}.parquet"
            else:
                pattern = f"{timeframe}_*.parquet"
            
            files = list(symbol_dir.glob(pattern))
            if not files:
                logger.warning(f"No feature files matching {pattern}")
                return None
            
            # Use most recent if multiple
            filepath = sorted(files)[-1]
            df = pd.read_parquet(filepath)
            
            logger.info(f"Loaded features for {symbol} from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return None
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of DataFrame."""
        # Use column names and shape for hash
        content = f"{list(df.columns)}_{df.shape}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_feature_list(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of available features.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of feature names
        """
        if category and category in self.feature_definitions:
            return self.feature_definitions[category]
        
        # Return all features
        all_features = []
        for features in self.feature_definitions.values():
            all_features.extend(features)
        return all_features
    
    # Specialized feature methods for ML pipeline
    
    def compute_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute intraday features on minute bars for mean reversion strategy.
        Expected columns: ['open','high','low','close','volume'] indexed by timestamp.
        Returns a DataFrame with features aligned to input index (no future leakage).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with intraday features
        """
        out = pd.DataFrame(index=df.index)
        
        # Price features
        out['close'] = df['close']
        out['ret_1'] = df['close'].pct_change()
        
        # Z-score of price over 20 bars
        out['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-9)
        
        # RSI(14)
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        out['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ATR(14) proxy from high-low range
        tr = (df['high'] - df['low']).abs()
        out['atr_14'] = tr.rolling(14).mean()
        
        # Volume z-score
        out['vol_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-9)
        
        # Spread proxy
        out['spread_proxy'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
        
        # Time-of-day bins (optional)
        out['minute'] = out.index.minute
        out['hour'] = out.index.hour
        
        return out.dropna()
    
    def compute_swing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute swing features on daily bars for trend following strategy.
        Expected columns: ['open','high','low','close','volume'] daily, index=timestamp.
        
        Args:
            df: DataFrame with daily OHLCV data
            
        Returns:
            DataFrame with swing features
        """
        out = pd.DataFrame(index=df.index)
        
        # Price features
        out['close'] = df['close']
        out['ema_20'] = df['close'].ewm(span=20).mean()
        out['ema_50'] = df['close'].ewm(span=50).mean()
        out['ema_20_slope'] = out['ema_20'].diff(5) / (out['ema_20'].shift(5) + 1e-9)
        
        # ADX proxy (simplified trend strength)
        out['trend_strength'] = (out['ema_20'] - out['ema_50']) / (out['ema_50'] + 1e-9)
        
        # ATR(14) daily
        tr = (df['high'] - df['low']).abs()
        out['atr_14'] = tr.rolling(14).mean()
        
        # Volume z-score
        out['vol_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-9)
        
        return out.dropna()
    
    def save_features_simple(self, symbol: str, timeframe: str, features: pd.DataFrame) -> str:
        """
        Save features to parquet in feature_store_dir/{timeframe}/{symbol}.parquet
        Simplified version for ML pipeline use.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe (e.g., '1Min', '1Day')
            features: DataFrame with features
            
        Returns:
            Path to saved file
        """
        out_dir = self.feature_dir / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{symbol}.parquet"
        features.to_parquet(path, index=True)
        logger.debug(f"Saved features for {symbol} to {path}")
        return str(path)
    
    def load_features_simple(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Load features from parquet.
        Simplified version for ML pipeline use.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe (e.g., '1Min', '1Day')
            
        Returns:
            DataFrame with features
        """
        path = self.feature_dir / timeframe / f"{symbol}.parquet"
        if not path.exists():
            logger.warning(f"No features found at {path}")
            return pd.DataFrame()
        return pd.read_parquet(path)
