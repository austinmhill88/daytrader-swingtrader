"""
Universe rotation cadence and change tracking (Phase 4).
Tracks symbol additions/removals and rotation metrics.
"""
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json


class UniverseRotationTracker:
    """
    Tracks universe changes and rotation metrics.
    Phase 4 implementation for portfolio orchestration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize universe rotation tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        storage_config = config.get('storage', {})
        
        # Storage
        self.data_dir = Path(storage_config.get('data_dir', './data'))
        self.rotation_file = self.data_dir / 'universe_rotation.json'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotation configuration
        self.rotation_frequency_days = 30  # Monthly rotation
        self.min_retention_days = 14  # Minimum days to keep a symbol
        
        # Tracking
        self.current_universe: Set[str] = set()
        self.previous_universe: Set[str] = set()
        self.rotation_history: List[Dict] = []
        self.symbol_metrics: Dict[str, Dict] = {}
        
        # Load existing data
        self._load_rotation_history()
        
        logger.info(
            f"UniverseRotationTracker initialized | "
            f"Rotation frequency: {self.rotation_frequency_days} days"
        )
    
    def update_universe(
        self,
        new_universe: Set[str],
        metrics: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        Update universe and track changes.
        
        Args:
            new_universe: New set of symbols
            metrics: Optional metrics per symbol (liquidity, volatility, etc.)
            
        Returns:
            Dictionary with rotation summary
        """
        now = datetime.now()
        
        # Track previous state
        self.previous_universe = self.current_universe.copy()
        
        # Calculate changes
        added = new_universe - self.previous_universe
        removed = self.previous_universe - new_universe
        retained = new_universe & self.previous_universe
        
        # Update current universe
        self.current_universe = new_universe
        
        # Update symbol metrics
        if metrics:
            for symbol, symbol_metrics in metrics.items():
                if symbol in new_universe:
                    if symbol not in self.symbol_metrics:
                        self.symbol_metrics[symbol] = {
                            'first_added': now.isoformat(),
                            'times_added': 1,
                            'times_removed': 0,
                            'total_days': 0
                        }
                    
                    # Update metrics
                    self.symbol_metrics[symbol].update(symbol_metrics)
                    self.symbol_metrics[symbol]['last_seen'] = now.isoformat()
        
        # Track removed symbols
        for symbol in removed:
            if symbol in self.symbol_metrics:
                self.symbol_metrics[symbol]['times_removed'] = (
                    self.symbol_metrics[symbol].get('times_removed', 0) + 1
                )
                self.symbol_metrics[symbol]['last_removed'] = now.isoformat()
        
        # Create rotation record
        rotation_record = {
            'timestamp': now.isoformat(),
            'universe_size': len(new_universe),
            'added': list(added),
            'removed': list(removed),
            'retained': len(retained),
            'retention_rate': (len(retained) / len(self.previous_universe) * 100) 
                if self.previous_universe else 100.0,
            'turnover': len(added) + len(removed)
        }
        
        # Add to history
        self.rotation_history.append(rotation_record)
        
        # Save to disk
        self._save_rotation_history()
        
        logger.info(
            f"Universe updated | "
            f"Size: {len(new_universe)}, "
            f"Added: {len(added)}, "
            f"Removed: {len(removed)}, "
            f"Retention: {rotation_record['retention_rate']:.1f}%"
        )
        
        if added:
            logger.debug(f"Added symbols: {', '.join(sorted(added))}")
        if removed:
            logger.debug(f"Removed symbols: {', '.join(sorted(removed))}")
        
        return rotation_record
    
    def should_rotate(self, last_rotation: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if universe should be rotated.
        
        Args:
            last_rotation: Timestamp of last rotation
            
        Returns:
            Tuple of (should_rotate, reason)
        """
        # Check if we have rotation history
        if not self.rotation_history:
            return True, "Initial universe setup"
        
        # Get last rotation time
        if last_rotation is None:
            last_record = self.rotation_history[-1]
            last_rotation = datetime.fromisoformat(last_record['timestamp'])
        
        # Check if enough time has passed
        days_since = (datetime.now() - last_rotation).days
        
        if days_since >= self.rotation_frequency_days:
            return True, f"Scheduled rotation ({days_since} days since last)"
        
        return False, f"Next rotation in {self.rotation_frequency_days - days_since} days"
    
    def get_symbol_tenure(self, symbol: str) -> int:
        """
        Get number of days a symbol has been in the universe.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Number of days
        """
        if symbol not in self.symbol_metrics:
            return 0
        
        first_added = self.symbol_metrics[symbol].get('first_added')
        if not first_added:
            return 0
        
        first_added_dt = datetime.fromisoformat(first_added)
        return (datetime.now() - first_added_dt).days
    
    def can_remove_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a symbol can be removed from universe.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (can_remove, reason)
        """
        tenure = self.get_symbol_tenure(symbol)
        
        if tenure < self.min_retention_days:
            return False, f"Minimum retention period not met ({tenure}/{self.min_retention_days} days)"
        
        return True, "Can be removed"
    
    def get_rotation_metrics(self, lookback_days: int = 90) -> Dict:
        """
        Get rotation metrics over a lookback period.
        
        Args:
            lookback_days: Days to look back
            
        Returns:
            Dictionary with rotation metrics
        """
        if not self.rotation_history:
            return {
                'rotations': 0,
                'avg_turnover': 0,
                'avg_retention_rate': 0,
                'total_symbols_seen': 0
            }
        
        # Filter to lookback period
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_rotations = [
            r for r in self.rotation_history
            if datetime.fromisoformat(r['timestamp']) >= cutoff
        ]
        
        if not recent_rotations:
            return {
                'rotations': 0,
                'avg_turnover': 0,
                'avg_retention_rate': 0,
                'total_symbols_seen': 0
            }
        
        # Calculate metrics
        turnovers = [r['turnover'] for r in recent_rotations]
        retention_rates = [r['retention_rate'] for r in recent_rotations]
        
        # Count unique symbols
        all_symbols = set()
        for r in recent_rotations:
            all_symbols.update(r['added'])
            all_symbols.update(r['removed'])
        
        return {
            'rotations': len(recent_rotations),
            'avg_turnover': sum(turnovers) / len(turnovers) if turnovers else 0,
            'avg_retention_rate': sum(retention_rates) / len(retention_rates) if retention_rates else 0,
            'total_symbols_seen': len(all_symbols),
            'current_universe_size': len(self.current_universe),
            'lookback_days': lookback_days
        }
    
    def get_symbol_stability_score(self, symbol: str) -> float:
        """
        Calculate stability score for a symbol (0-1).
        Higher score = more stable/retained in universe.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stability score
        """
        if symbol not in self.symbol_metrics:
            return 0.0
        
        metrics = self.symbol_metrics[symbol]
        
        # Factors:
        # 1. Tenure (longer = more stable)
        tenure_days = self.get_symbol_tenure(symbol)
        tenure_score = min(tenure_days / 90.0, 1.0)  # Cap at 90 days
        
        # 2. Add/remove ratio (fewer removes = more stable)
        times_added = metrics.get('times_added', 1)
        times_removed = metrics.get('times_removed', 0)
        
        if times_added > 0:
            retention_ratio = 1.0 - (times_removed / (times_added + times_removed))
        else:
            retention_ratio = 0.0
        
        # Weighted average
        stability_score = (tenure_score * 0.6) + (retention_ratio * 0.4)
        
        return stability_score
    
    def get_top_stable_symbols(self, n: int = 50) -> List[Tuple[str, float]]:
        """
        Get top N most stable symbols.
        
        Args:
            n: Number of symbols to return
            
        Returns:
            List of (symbol, stability_score) tuples
        """
        scores = [
            (symbol, self.get_symbol_stability_score(symbol))
            for symbol in self.current_universe
        ]
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:n]
    
    def _save_rotation_history(self) -> None:
        """Save rotation history to disk."""
        try:
            data = {
                'rotation_history': self.rotation_history[-100:],  # Keep last 100 rotations
                'symbol_metrics': self.symbol_metrics,
                'current_universe': list(self.current_universe),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.rotation_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving rotation history: {e}")
    
    def _load_rotation_history(self) -> None:
        """Load rotation history from disk."""
        try:
            if not self.rotation_file.exists():
                logger.info("No existing rotation history found")
                return
            
            with open(self.rotation_file, 'r') as f:
                data = json.load(f)
            
            self.rotation_history = data.get('rotation_history', [])
            self.symbol_metrics = data.get('symbol_metrics', {})
            self.current_universe = set(data.get('current_universe', []))
            
            logger.info(
                f"Loaded rotation history: "
                f"{len(self.rotation_history)} rotations, "
                f"{len(self.symbol_metrics)} symbols tracked"
            )
            
        except Exception as e:
            logger.error(f"Error loading rotation history: {e}")
    
    def get_universe_summary(self) -> Dict:
        """
        Get comprehensive universe summary.
        
        Returns:
            Summary dictionary
        """
        rotation_metrics = self.get_rotation_metrics(90)
        top_stable = self.get_top_stable_symbols(10)
        
        return {
            'current_size': len(self.current_universe),
            'total_tracked': len(self.symbol_metrics),
            'rotation_metrics': rotation_metrics,
            'top_stable_symbols': [
                {'symbol': s, 'stability': score}
                for s, score in top_stable
            ],
            'avg_symbol_tenure_days': (
                sum(self.get_symbol_tenure(s) for s in self.current_universe) / 
                len(self.current_universe)
            ) if self.current_universe else 0
        }
