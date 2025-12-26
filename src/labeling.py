"""
Triple-barrier labeling and meta-labeling for ML model training.
Implements sophisticated labeling methods to reduce label noise and align with actual exits.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class TripleBarrierLabeler:
    """
    Triple-barrier labeling system for supervised ML.
    Creates labels based on first touch of profit target, stop loss, or time exit.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize triple-barrier labeler.
        
        Args:
            config: Configuration dictionary with labeling parameters
        """
        self.config = config
        labeling_config = config.get('ml_training', {}).get('labeling', {})
        
        # Barrier thresholds (in ATR multiples)
        self.profit_target_atr = labeling_config.get('profit_target_atr', 1.5)
        self.stop_loss_atr = labeling_config.get('stop_loss_atr', 1.0)
        
        # Time-based exit (in bars)
        self.max_holding_bars = labeling_config.get('max_holding_bars', 30)
        
        # Meta-labeling parameters
        self.enable_meta_labeling = labeling_config.get('enable_meta_labeling', True)
        
        logger.info(
            f"TripleBarrierLabeler initialized | "
            f"Profit: {self.profit_target_atr}x ATR, "
            f"Stop: {self.stop_loss_atr}x ATR, "
            f"Max hold: {self.max_holding_bars} bars"
        )
    
    def apply_triple_barrier(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        atr_col: str = 'atr_14'
    ) -> pd.DataFrame:
        """
        Apply triple-barrier labeling to a DataFrame.
        
        Args:
            df: DataFrame with price and ATR data
            price_col: Name of price column
            atr_col: Name of ATR column
            
        Returns:
            DataFrame with added label columns:
            - barrier_touched: which barrier was hit first (1=profit, -1=stop, 0=time)
            - return_at_exit: actual return at barrier touch
            - holding_period: number of bars held
            - label: binary label for ML (1 if profitable exit)
        """
        df = df.copy()
        
        # Initialize output columns
        df['barrier_touched'] = 0
        df['return_at_exit'] = 0.0
        df['holding_period'] = 0
        df['label'] = 0
        
        # Process each entry point
        for idx in range(len(df) - self.max_holding_bars):
            entry_price = df[price_col].iloc[idx]
            atr_value = df[atr_col].iloc[idx]
            
            if pd.isna(entry_price) or pd.isna(atr_value) or atr_value == 0:
                continue
            
            # Calculate barriers
            profit_target = entry_price + (atr_value * self.profit_target_atr)
            stop_loss = entry_price - (atr_value * self.stop_loss_atr)
            
            # Look forward to find first barrier touch
            for forward_idx in range(1, self.max_holding_bars + 1):
                if idx + forward_idx >= len(df):
                    break
                
                future_price = df[price_col].iloc[idx + forward_idx]
                
                if pd.isna(future_price):
                    break
                
                # Check profit target
                if future_price >= profit_target:
                    df.loc[df.index[idx], 'barrier_touched'] = 1
                    df.loc[df.index[idx], 'return_at_exit'] = (future_price - entry_price) / entry_price
                    df.loc[df.index[idx], 'holding_period'] = forward_idx
                    df.loc[df.index[idx], 'label'] = 1
                    break
                
                # Check stop loss
                elif future_price <= stop_loss:
                    df.loc[df.index[idx], 'barrier_touched'] = -1
                    df.loc[df.index[idx], 'return_at_exit'] = (future_price - entry_price) / entry_price
                    df.loc[df.index[idx], 'holding_period'] = forward_idx
                    df.loc[df.index[idx], 'label'] = 0
                    break
                
                # Time exit (last iteration)
                elif forward_idx == self.max_holding_bars:
                    df.loc[df.index[idx], 'barrier_touched'] = 0
                    df.loc[df.index[idx], 'return_at_exit'] = (future_price - entry_price) / entry_price
                    df.loc[df.index[idx], 'holding_period'] = forward_idx
                    # Label based on profitability
                    df.loc[df.index[idx], 'label'] = 1 if future_price > entry_price else 0
        
        logger.info(
            f"Triple-barrier labeling complete | "
            f"{len(df)} samples, "
            f"Profit: {(df['barrier_touched'] == 1).sum()}, "
            f"Stop: {(df['barrier_touched'] == -1).sum()}, "
            f"Time: {(df['barrier_touched'] == 0).sum()}"
        )
        
        return df
    
    def create_meta_labels(
        self,
        df: pd.DataFrame,
        primary_signal_col: str = 'signal'
    ) -> pd.DataFrame:
        """
        Create meta-labels for two-stage modeling.
        Stage 1: Should we trade? (filter model)
        Stage 2: What direction/size? (direction model)
        
        Args:
            df: DataFrame with barrier labels
            primary_signal_col: Column name of primary trading signal
            
        Returns:
            DataFrame with meta-label columns:
            - meta_label_filter: 1 if should trade, 0 if should skip
            - meta_label_direction: 1 for long, -1 for short, 0 for skip
        """
        df = df.copy()
        
        # Filter meta-label: trade if hit profit before stop or time exit is profitable
        df['meta_label_filter'] = (
            (df['barrier_touched'] == 1) |  # Hit profit target
            ((df['barrier_touched'] == 0) & (df['return_at_exit'] > 0))  # Time exit profitable
        ).astype(int)
        
        # Direction meta-label: align with primary signal if should trade
        if primary_signal_col in df.columns:
            df['meta_label_direction'] = np.where(
                df['meta_label_filter'] == 1,
                df[primary_signal_col],
                0
            )
        else:
            # Default to label-based direction
            df['meta_label_direction'] = np.where(
                df['meta_label_filter'] == 1,
                df['label'] * 2 - 1,  # Convert 0/1 to -1/1
                0
            )
        
        logger.info(
            f"Meta-labeling complete | "
            f"Trade rate: {df['meta_label_filter'].mean():.2%}, "
            f"Positive labels: {(df['label'] == 1).sum()}"
        )
        
        return df


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series with embargo.
    Prevents leakage by purging training samples that are too close to test samples.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_window: int = 5
    ):
        """
        Initialize purged k-fold.
        
        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of data to embargo after test set
            purge_window: Number of samples to purge before test set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_window = purge_window
        
        logger.info(
            f"PurgedKFold initialized | "
            f"n_splits={n_splits}, "
            f"embargo_pct={embargo_pct}, "
            f"purge_window={purge_window}"
        )
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target series (optional)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        test_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        indices = np.arange(n_samples)
        splits = []
        
        for fold in range(self.n_splits):
            # Calculate test set boundaries
            test_start = fold * test_size
            test_end = test_start + test_size if fold < self.n_splits - 1 else n_samples
            
            # Define test set
            test_indices = indices[test_start:test_end]
            
            # Define train set with purging and embargo
            # Purge: remove samples before test set
            purge_start = max(0, test_start - self.purge_window)
            
            # Embargo: remove samples after test set
            embargo_end = min(n_samples, test_end + embargo_size)
            
            # Train set excludes purged, test, and embargoed regions
            train_indices = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])
            
            splits.append((train_indices, test_indices))
            
            logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"Train size: {len(train_indices)}, "
                f"Test size: {len(test_indices)}, "
                f"Purged: {purge_start} to {test_start}, "
                f"Embargoed: {test_end} to {embargo_end}"
            )
        
        return splits
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


class SampleWeights:
    """
    Calculate sample weights for ML training to handle:
    - Overlapping samples
    - Class imbalance
    - Return-based weighting
    """
    
    @staticmethod
    def calculate_return_weights(
        returns: pd.Series,
        method: str = 'abs'
    ) -> pd.Series:
        """
        Calculate weights based on returns.
        
        Args:
            returns: Series of returns
            method: 'abs' for absolute value, 'sign' for sign-based
            
        Returns:
            Series of weights
        """
        if method == 'abs':
            # Weight by absolute return magnitude
            weights = np.abs(returns)
        elif method == 'sign':
            # Weight by return direction strength
            weights = np.abs(returns) * np.sign(returns)
        else:
            weights = pd.Series(1.0, index=returns.index)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    @staticmethod
    def calculate_time_decay_weights(
        df: pd.DataFrame,
        half_life: int = 100
    ) -> pd.Series:
        """
        Calculate exponentially decaying weights by time.
        More recent samples get higher weight.
        
        Args:
            df: DataFrame with time-indexed data
            half_life: Half-life for exponential decay (in samples)
            
        Returns:
            Series of weights
        """
        n = len(df)
        # Calculate decay factor
        decay = np.exp(-np.log(2) / half_life)
        
        # Generate weights (most recent = highest)
        weights = pd.Series([decay ** (n - i - 1) for i in range(n)], index=df.index)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    @staticmethod
    def calculate_class_balance_weights(
        y: pd.Series
    ) -> pd.Series:
        """
        Calculate weights to balance classes.
        
        Args:
            y: Target series
            
        Returns:
            Series of weights
        """
        # Count classes
        class_counts = y.value_counts()
        total = len(y)
        
        # Calculate weight per class (inverse frequency)
        class_weights = {cls: total / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        
        # Apply to samples
        weights = y.map(class_weights)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    @staticmethod
    def calculate_combined_weights(
        df: pd.DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None,
        time_decay_half_life: int = 100
    ) -> pd.Series:
        """
        Calculate combined weights from multiple sources.
        
        Args:
            df: Feature DataFrame
            y: Target series
            returns: Optional returns series for return-based weighting
            time_decay_half_life: Half-life for time decay
            
        Returns:
            Series of combined weights
        """
        # Start with uniform weights
        weights = pd.Series(1.0, index=df.index)
        
        # Apply class balance weighting
        class_weights = SampleWeights.calculate_class_balance_weights(y)
        weights = weights * class_weights
        
        # Apply time decay weighting
        time_weights = SampleWeights.calculate_time_decay_weights(df, time_decay_half_life)
        weights = weights * time_weights
        
        # Apply return weighting if available
        if returns is not None:
            return_weights = SampleWeights.calculate_return_weights(returns)
            weights = weights * return_weights
        
        # Normalize
        weights = weights / weights.sum()
        
        logger.info(
            f"Combined weights calculated | "
            f"Min: {weights.min():.6f}, "
            f"Max: {weights.max():.6f}, "
            f"Mean: {weights.mean():.6f}"
        )
        
        return weights


def apply_sample_uniqueness(
    df: pd.DataFrame,
    holding_periods: pd.Series
) -> pd.Series:
    """
    Calculate sample uniqueness based on overlapping holding periods.
    Reduces the weight of samples that overlap with many other samples.
    
    Args:
        df: Feature DataFrame (time-indexed)
        holding_periods: Series of holding periods for each sample
        
    Returns:
        Series of uniqueness scores (0-1)
    """
    n = len(df)
    uniqueness = pd.Series(1.0, index=df.index)
    
    # For each sample, count overlaps with other samples
    for i, idx in enumerate(df.index):
        if i >= len(holding_periods):
            break
            
        holding = holding_periods.iloc[i]
        if pd.isna(holding) or holding == 0:
            continue
        
        # Time range of this sample's holding period
        start_time = idx
        end_time_idx = min(i + int(holding), n - 1)
        
        # Count overlapping samples
        overlap_count = 0
        for j in range(max(0, i - int(holding)), min(n, i + int(holding))):
            if j != i:
                other_holding = holding_periods.iloc[j]
                if not pd.isna(other_holding) and other_holding > 0:
                    # Check if periods overlap
                    other_end = min(j + int(other_holding), n - 1)
                    if not (end_time_idx < j or i > other_end):
                        overlap_count += 1
        
        # Calculate uniqueness (inverse of overlap)
        uniqueness.iloc[i] = 1.0 / (1.0 + overlap_count)
    
    logger.info(
        f"Sample uniqueness calculated | "
        f"Mean: {uniqueness.mean():.3f}, "
        f"Min: {uniqueness.min():.3f}"
    )
    
    return uniqueness
