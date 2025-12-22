"""
Ensemble strategy selector for dynamic strategy allocation.
Selects optimal subset of strategies based on regime, liquidity, and performance.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np


@dataclass
class StrategyScore:
    """Score for a strategy candidate."""
    name: str
    score: float
    regime_fit: float
    recent_performance: float
    capacity: float
    risk_budget_pct: float
    reason: str


class EnsembleSelector:
    """
    Select optimal ensemble of strategies based on current market conditions.
    Dynamically allocates risk budget across strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ensemble selector.
        
        Args:
            config: Application configuration
        """
        self.config = config
        ensemble_config = config.get('ensemble', {})
        
        self.min_strategies = ensemble_config.get('min_strategies', 5)
        self.max_strategies = ensemble_config.get('max_strategies', 15)
        self.lookback_days = ensemble_config.get('lookback_days', 30)
        self.min_sharpe = ensemble_config.get('min_sharpe', 0.5)
        self.max_correlation = ensemble_config.get('max_correlation', 0.7)
        self.underperformer_threshold_days = ensemble_config.get('underperformer_threshold_days', 7)
        
        # Performance tracking
        self.strategy_performance: Dict[str, List[Dict]] = {}
        self.strategy_enabled: Dict[str, bool] = {}
        
        logger.info(
            f"EnsembleSelector initialized | "
            f"Range: {self.min_strategies}-{self.max_strategies} strategies"
        )
    
    def score_strategy(
        self,
        strategy_name: str,
        regime: str,
        market_volatility: float,
        liquidity_score: float,
        recent_trades: Optional[List[Dict]] = None
    ) -> StrategyScore:
        """
        Score a strategy candidate based on multiple factors.
        
        Args:
            strategy_name: Strategy name
            regime: Current market regime
            market_volatility: Current volatility level
            liquidity_score: Market liquidity score
            recent_trades: Recent trade history for the strategy
            
        Returns:
            StrategyScore object
        """
        # Initialize scores
        regime_fit = 0.0
        recent_performance = 0.0
        capacity = 1.0
        
        # 1. Regime fitness score
        regime_fit = self._calculate_regime_fit(strategy_name, regime, market_volatility)
        
        # 2. Recent performance score
        if recent_trades:
            recent_performance = self._calculate_recent_performance(recent_trades)
        
        # 3. Capacity score based on liquidity
        capacity = self._calculate_capacity_score(strategy_name, liquidity_score)
        
        # 4. Combined score (weighted)
        combined_score = (
            0.4 * regime_fit +
            0.4 * recent_performance +
            0.2 * capacity
        )
        
        # 5. Risk budget allocation
        risk_budget_pct = self._allocate_risk_budget(
            combined_score,
            strategy_name,
            recent_performance
        )
        
        # Generate reason
        reason = self._generate_selection_reason(
            regime_fit, recent_performance, capacity
        )
        
        return StrategyScore(
            name=strategy_name,
            score=combined_score,
            regime_fit=regime_fit,
            recent_performance=recent_performance,
            capacity=capacity,
            risk_budget_pct=risk_budget_pct,
            reason=reason
        )
    
    def _calculate_regime_fit(
        self,
        strategy_name: str,
        regime: str,
        volatility: float
    ) -> float:
        """
        Calculate how well a strategy fits current market regime.
        
        Args:
            strategy_name: Strategy name
            regime: Current regime
            volatility: Current volatility
            
        Returns:
            Fit score 0-1
        """
        # Define strategy preferences for different regimes
        regime_preferences = {
            'trending': {
                'swing_trend_following': 0.9,
                'intraday_mean_reversion': 0.4,
                'momentum': 0.8,
                'breakout': 0.85
            },
            'choppy': {
                'swing_trend_following': 0.3,
                'intraday_mean_reversion': 0.9,
                'momentum': 0.4,
                'range_trading': 0.85
            },
            'high_volatility': {
                'swing_trend_following': 0.6,
                'intraday_mean_reversion': 0.7,
                'momentum': 0.5,
                'volatility_arbitrage': 0.9
            },
            'low_volatility': {
                'swing_trend_following': 0.7,
                'intraday_mean_reversion': 0.6,
                'momentum': 0.7,
                'carry': 0.8
            }
        }
        
        # Get fit score for this regime
        regime_scores = regime_preferences.get(regime, {})
        base_score = regime_scores.get(strategy_name, 0.5)  # Default to neutral
        
        # Adjust for volatility
        if 'mean_reversion' in strategy_name.lower():
            # Mean reversion does better in moderate volatility
            if 0.15 < volatility < 0.35:
                base_score *= 1.2
        elif 'trend' in strategy_name.lower():
            # Trend following does better in trending markets
            if volatility > 0.25:
                base_score *= 1.1
        
        return min(1.0, base_score)
    
    def _calculate_recent_performance(
        self,
        trades: List[Dict]
    ) -> float:
        """
        Calculate recent performance score from trade history.
        
        Args:
            trades: Recent trades
            
        Returns:
            Performance score 0-1
        """
        if not trades:
            return 0.5  # Neutral if no history
        
        # Calculate win rate
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / len(trades) if trades else 0.5
        
        # Calculate average profit factor
        total_wins = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        total_losses = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0
        
        # Combined score
        score = (0.5 * win_rate) + (0.5 * min(1.0, profit_factor / 2.0))
        
        return score
    
    def _calculate_capacity_score(
        self,
        strategy_name: str,
        liquidity_score: float
    ) -> float:
        """
        Calculate strategy capacity based on market liquidity.
        
        Args:
            strategy_name: Strategy name
            liquidity_score: Current liquidity score
            
        Returns:
            Capacity score 0-1
        """
        # High-frequency strategies need more liquidity
        if 'intraday' in strategy_name.lower() or 'scalping' in strategy_name.lower():
            # Require higher liquidity
            return min(1.0, liquidity_score / 0.8)
        else:
            # Swing/position strategies are less liquidity-sensitive
            return min(1.0, liquidity_score / 0.5)
    
    def _allocate_risk_budget(
        self,
        score: float,
        strategy_name: str,
        performance: float
    ) -> float:
        """
        Allocate risk budget percentage to a strategy.
        
        Args:
            score: Combined strategy score
            strategy_name: Strategy name
            performance: Recent performance score
            
        Returns:
            Risk budget percentage
        """
        # Base allocation proportional to score
        base_allocation = score * 10.0  # Scale to percentage
        
        # Adjust based on recent performance
        if performance > 0.6:
            base_allocation *= 1.2  # Boost high performers
        elif performance < 0.4:
            base_allocation *= 0.8  # Reduce underperformers
        
        # Ensure reasonable range
        return min(15.0, max(2.0, base_allocation))
    
    def _generate_selection_reason(
        self,
        regime_fit: float,
        performance: float,
        capacity: float
    ) -> str:
        """
        Generate human-readable reason for strategy selection.
        
        Args:
            regime_fit: Regime fit score
            performance: Performance score
            capacity: Capacity score
            
        Returns:
            Reason string
        """
        reasons = []
        
        if regime_fit > 0.7:
            reasons.append("strong regime fit")
        elif regime_fit < 0.3:
            reasons.append("weak regime fit")
        
        if performance > 0.6:
            reasons.append("good recent performance")
        elif performance < 0.4:
            reasons.append("weak recent performance")
        
        if capacity > 0.8:
            reasons.append("high capacity")
        elif capacity < 0.5:
            reasons.append("limited capacity")
        
        if not reasons:
            return "balanced metrics"
        
        return ", ".join(reasons)
    
    def select_ensemble(
        self,
        available_strategies: List[str],
        regime: str,
        market_volatility: float,
        liquidity_score: float,
        strategy_trades: Optional[Dict[str, List[Dict]]] = None
    ) -> List[StrategyScore]:
        """
        Select optimal ensemble of strategies.
        
        Args:
            available_strategies: List of available strategy names
            regime: Current market regime
            market_volatility: Current volatility
            liquidity_score: Market liquidity score
            strategy_trades: Recent trades by strategy
            
        Returns:
            List of selected StrategyScore objects
        """
        if strategy_trades is None:
            strategy_trades = {}
        
        # Score all strategies
        scored_strategies = []
        for strategy_name in available_strategies:
            recent_trades = strategy_trades.get(strategy_name, [])
            score = self.score_strategy(
                strategy_name,
                regime,
                market_volatility,
                liquidity_score,
                recent_trades
            )
            scored_strategies.append(score)
        
        # Sort by score descending
        scored_strategies.sort(key=lambda x: x.score, reverse=True)
        
        # Select top N strategies
        num_to_select = min(self.max_strategies, max(self.min_strategies, len(scored_strategies)))
        selected = scored_strategies[:num_to_select]
        
        # Log selection
        logger.info(
            f"Ensemble selected | {len(selected)} strategies | "
            f"Regime: {regime}, Volatility: {market_volatility:.2%}"
        )
        
        for strategy in selected:
            logger.info(
                f"  • {strategy.name} | "
                f"Score: {strategy.score:.2f}, "
                f"Risk Budget: {strategy.risk_budget_pct:.1f}%, "
                f"Reason: {strategy.reason}"
            )
        
        return selected
    
    def check_underperformers(
        self,
        strategy_trades: Dict[str, List[Dict]]
    ) -> List[str]:
        """
        Identify chronic underperforming strategies.
        
        Args:
            strategy_trades: Recent trades by strategy
            
        Returns:
            List of underperformer strategy names
        """
        underperformers = []
        cutoff_date = datetime.now() - timedelta(days=self.underperformer_threshold_days)
        
        for strategy_name, trades in strategy_trades.items():
            # Filter to recent trades
            recent_trades = [
                t for t in trades
                if t.get('timestamp', datetime.min) > cutoff_date
            ]
            
            if not recent_trades or len(recent_trades) < 5:
                continue  # Not enough data
            
            # Calculate metrics
            total_pnl = sum(t.get('pnl', 0) for t in recent_trades)
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            win_rate = wins / len(recent_trades)
            
            # Check if underperforming
            if total_pnl < 0 and win_rate < 0.4:
                underperformers.append(strategy_name)
                logger.warning(
                    f"Underperformer detected: {strategy_name} | "
                    f"P&L: ${total_pnl:.2f}, Win rate: {win_rate:.1%}"
                )
        
        return underperformers
