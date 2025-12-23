"""
Capital allocation across strategies (Phase 4).
Implements Kelly criterion and volatility targeting for portfolio optimization.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class CapitalAllocator:
    """
    Capital allocation manager for multiple strategies.
    Phase 4 implementation for portfolio orchestration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize capital allocator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Strategy configurations
        self.strategies = {}
        strategy_configs = config.get('strategies', {})
        
        for strategy_name, strategy_config in strategy_configs.items():
            if strategy_config.get('enabled', False):
                self.strategies[strategy_name] = {
                    'max_positions': strategy_config.get('max_positions', 10),
                    'enabled': True,
                    'type': 'intraday' if 'intraday' in strategy_name else 'swing'
                }
        
        # Allocation method
        self.allocation_method = 'equal_weight'  # 'equal_weight', 'risk_parity', 'kelly', 'vol_target'
        
        # Volatility targeting
        self.target_portfolio_vol = 0.15  # 15% annualized
        self.vol_lookback_days = 60
        
        # Risk limits per strategy type
        self.max_allocation_pct = {
            'intraday': 50.0,
            'swing': 100.0
        }
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict] = {}
        
        logger.info(
            f"CapitalAllocator initialized | "
            f"Strategies: {len(self.strategies)}, "
            f"Method: {self.allocation_method}"
        )
    
    def allocate_capital(
        self,
        total_equity: float,
        strategy_metrics: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Allocate capital across strategies.
        
        Args:
            total_equity: Total account equity
            strategy_metrics: Dict of strategy -> performance metrics
            
        Returns:
            Dict of strategy -> allocated capital
        """
        if self.allocation_method == 'equal_weight':
            return self._equal_weight_allocation(total_equity)
        
        elif self.allocation_method == 'risk_parity':
            return self._risk_parity_allocation(total_equity, strategy_metrics)
        
        elif self.allocation_method == 'kelly':
            return self._kelly_allocation(total_equity, strategy_metrics)
        
        elif self.allocation_method == 'vol_target':
            return self._vol_target_allocation(total_equity, strategy_metrics)
        
        else:
            logger.warning(f"Unknown allocation method: {self.allocation_method}")
            return self._equal_weight_allocation(total_equity)
    
    def _equal_weight_allocation(self, total_equity: float) -> Dict[str, float]:
        """
        Allocate capital equally across all strategies.
        
        Args:
            total_equity: Total equity
            
        Returns:
            Dict of strategy -> allocation
        """
        if not self.strategies:
            return {}
        
        # Apply strategy type limits
        allocations = {}
        n_strategies = len(self.strategies)
        
        for strategy_name, strategy_config in self.strategies.items():
            strategy_type = strategy_config['type']
            max_alloc = self.max_allocation_pct[strategy_type] / 100.0
            
            # Equal split, capped by strategy type limit
            equal_share = 1.0 / n_strategies
            allocation_pct = min(equal_share, max_alloc)
            
            allocations[strategy_name] = total_equity * allocation_pct
        
        logger.debug(
            f"Equal weight allocation: "
            f"{', '.join([f'{k}: ${v:,.0f}' for k, v in allocations.items()])}"
        )
        
        return allocations
    
    def _risk_parity_allocation(
        self,
        total_equity: float,
        strategy_metrics: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Allocate capital using risk parity (equal risk contribution).
        
        Args:
            total_equity: Total equity
            strategy_metrics: Performance metrics per strategy
            
        Returns:
            Dict of strategy -> allocation
        """
        # Calculate inverse volatility weights
        inv_vols = {}
        
        for strategy_name in self.strategies.keys():
            metrics = strategy_metrics.get(strategy_name, {})
            vol = metrics.get('volatility', 0.15)  # Default 15% if not available
            
            if vol > 0:
                inv_vols[strategy_name] = 1.0 / vol
            else:
                inv_vols[strategy_name] = 0
        
        # Normalize to sum to 1
        total_inv_vol = sum(inv_vols.values())
        
        allocations = {}
        for strategy_name, inv_vol in inv_vols.items():
            if total_inv_vol > 0:
                weight = inv_vol / total_inv_vol
            else:
                weight = 1.0 / len(self.strategies)
            
            # Apply strategy type limits
            strategy_type = self.strategies[strategy_name]['type']
            max_alloc = self.max_allocation_pct[strategy_type] / 100.0
            weight = min(weight, max_alloc)
            
            allocations[strategy_name] = total_equity * weight
        
        logger.debug(
            f"Risk parity allocation: "
            f"{', '.join([f'{k}: ${v:,.0f}' for k, v in allocations.items()])}"
        )
        
        return allocations
    
    def _kelly_allocation(
        self,
        total_equity: float,
        strategy_metrics: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Allocate capital using Kelly criterion.
        
        Args:
            total_equity: Total equity
            strategy_metrics: Performance metrics per strategy
            
        Returns:
            Dict of strategy -> allocation
        """
        kelly_fractions = {}
        
        for strategy_name in self.strategies.keys():
            metrics = strategy_metrics.get(strategy_name, {})
            
            win_rate = metrics.get('win_rate', 0.5)
            avg_win = metrics.get('avg_win', 0.02)
            avg_loss = metrics.get('avg_loss', 0.01)
            
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly_f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            else:
                kelly_f = 0
            
            # Use fractional Kelly (e.g., half-Kelly)
            kelly_fractions[strategy_name] = max(0, kelly_f * 0.5)
        
        # Normalize
        total_kelly = sum(kelly_fractions.values())
        
        allocations = {}
        for strategy_name, kelly_f in kelly_fractions.items():
            if total_kelly > 0:
                weight = kelly_f / total_kelly
            else:
                weight = 1.0 / len(self.strategies)
            
            # Apply strategy type limits
            strategy_type = self.strategies[strategy_name]['type']
            max_alloc = self.max_allocation_pct[strategy_type] / 100.0
            weight = min(weight, max_alloc)
            
            allocations[strategy_name] = total_equity * weight
        
        logger.debug(
            f"Kelly allocation: "
            f"{', '.join([f'{k}: ${v:,.0f}' for k, v in allocations.items()])}"
        )
        
        return allocations
    
    def _vol_target_allocation(
        self,
        total_equity: float,
        strategy_metrics: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Allocate capital using volatility targeting.
        
        Args:
            total_equity: Total equity
            strategy_metrics: Performance metrics per strategy
            
        Returns:
            Dict of strategy -> allocation
        """
        vol_scalers = {}
        
        for strategy_name in self.strategies.keys():
            metrics = strategy_metrics.get(strategy_name, {})
            vol = metrics.get('volatility', 0.15)
            
            # Scale inversely with volatility to target portfolio vol
            if vol > 0:
                scaler = self.target_portfolio_vol / vol
            else:
                scaler = 1.0
            
            vol_scalers[strategy_name] = scaler
        
        # Normalize
        total_scaler = sum(vol_scalers.values())
        
        allocations = {}
        for strategy_name, scaler in vol_scalers.items():
            if total_scaler > 0:
                weight = scaler / total_scaler
            else:
                weight = 1.0 / len(self.strategies)
            
            # Apply strategy type limits
            strategy_type = self.strategies[strategy_name]['type']
            max_alloc = self.max_allocation_pct[strategy_type] / 100.0
            weight = min(weight, max_alloc)
            
            allocations[strategy_name] = total_equity * weight
        
        logger.debug(
            f"Vol target allocation: "
            f"{', '.join([f'{k}: ${v:,.0f}' for k, v in allocations.items()])}"
        )
        
        return allocations
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        returns: pd.Series,
        trades: List[Dict]
    ) -> None:
        """
        Update performance tracking for a strategy.
        
        Args:
            strategy_name: Strategy identifier
            returns: Series of returns
            trades: List of trade dicts
        """
        if strategy_name not in self.strategies:
            return
        
        # Calculate metrics
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            volatility = 0
            sharpe = 0
        
        # Calculate win rate and avg win/loss
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        self.strategy_performance[strategy_name] = {
            'volatility': volatility,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': len(trades),
            'last_updated': datetime.now().isoformat()
        }
        
        logger.debug(
            f"Updated performance for {strategy_name}: "
            f"Vol: {volatility:.2%}, Sharpe: {sharpe:.2f}, "
            f"Win rate: {win_rate:.2%}"
        )
    
    def get_allocation_summary(
        self,
        allocations: Dict[str, float],
        total_equity: float
    ) -> Dict:
        """
        Get summary of capital allocations.
        
        Args:
            allocations: Dict of strategy -> capital
            total_equity: Total equity
            
        Returns:
            Summary dictionary
        """
        total_allocated = sum(allocations.values())
        
        return {
            'total_equity': total_equity,
            'total_allocated': total_allocated,
            'allocation_pct': (total_allocated / total_equity * 100) if total_equity > 0 else 0,
            'allocations': allocations,
            'allocation_pcts': {
                k: (v / total_equity * 100) if total_equity > 0 else 0
                for k, v in allocations.items()
            },
            'method': self.allocation_method,
            'num_strategies': len(allocations)
        }
    
    def rebalance_needed(
        self,
        current_allocations: Dict[str, float],
        target_allocations: Dict[str, float],
        threshold_pct: float = 10.0
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if rebalancing is needed.
        
        Args:
            current_allocations: Current capital per strategy
            target_allocations: Target capital per strategy
            threshold_pct: Rebalance if deviation > threshold
            
        Returns:
            Tuple of (needs_rebalance, adjustments)
        """
        needs_rebalance = False
        adjustments = {}
        
        for strategy_name in target_allocations.keys():
            current = current_allocations.get(strategy_name, 0)
            target = target_allocations[strategy_name]
            
            if target > 0:
                deviation_pct = abs(current - target) / target * 100
                
                if deviation_pct > threshold_pct:
                    needs_rebalance = True
                    adjustments[strategy_name] = target - current
        
        return needs_rebalance, adjustments
    
    def set_allocation_method(self, method: str) -> bool:
        """
        Set capital allocation method.
        
        Args:
            method: Allocation method name
            
        Returns:
            True if valid method
        """
        valid_methods = ['equal_weight', 'risk_parity', 'kelly', 'vol_target']
        
        if method in valid_methods:
            self.allocation_method = method
            logger.info(f"Allocation method set to: {method}")
            return True
        else:
            logger.error(f"Invalid allocation method: {method}")
            return False
