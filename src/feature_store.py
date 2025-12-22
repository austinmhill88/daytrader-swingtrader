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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature store.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.feature_dir = Path(config.get('storage', {}).get('feature_store_dir', './data/features'))
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
