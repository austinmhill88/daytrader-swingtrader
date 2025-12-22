"""
Enhanced metrics tracking (Phase 3).
Tracks latency, fill quality, slippage, and system performance.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from loguru import logger


@dataclass
class LatencyMetric:
    """Latency measurement."""
    component: str
    operation: str
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FillMetric:
    """Order fill quality metric."""
    symbol: str
    order_type: str
    expected_price: float
    fill_price: float
    slippage_bps: float
    partial_fill: bool
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsTracker:
    """
    Enhanced metrics tracking for system performance.
    Phase 3 implementation for operational insights.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize metrics tracker.
        
        Args:
            config: Application configuration
        """
        self.config = config
        metrics_config = config.get('metrics', {})
        
        self.track_latency = metrics_config.get('track_latency', True)
        self.track_fill_quality = metrics_config.get('track_fill_quality', True)
        self.track_slippage = metrics_config.get('track_slippage', True)
        
        # Metric storage (keep last 10000 entries)
        self.latency_metrics: deque = deque(maxlen=10000)
        self.fill_metrics: deque = deque(maxlen=10000)
        
        # Aggregated stats
        self.latency_stats: Dict[str, Dict] = {}
        self.fill_stats: Dict[str, Dict] = {}
        
        logger.info("MetricsTracker initialized")
    
    def record_latency(
        self,
        component: str,
        operation: str,
        latency_ms: float
    ) -> None:
        """
        Record latency measurement.
        
        Args:
            component: Component name
            operation: Operation name
            latency_ms: Latency in milliseconds
        """
        if not self.track_latency:
            return
        
        metric = LatencyMetric(
            component=component,
            operation=operation,
            latency_ms=latency_ms
        )
        self.latency_metrics.append(metric)
        
        # Update stats
        key = f"{component}.{operation}"
        if key not in self.latency_stats:
            self.latency_stats[key] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': 0.0
            }
        
        stats = self.latency_stats[key]
        stats['count'] += 1
        stats['sum'] += latency_ms
        stats['min'] = min(stats['min'], latency_ms)
        stats['max'] = max(stats['max'], latency_ms)
    
    def record_fill(
        self,
        symbol: str,
        order_type: str,
        expected_price: float,
        fill_price: float,
        partial_fill: bool = False
    ) -> None:
        """
        Record order fill quality.
        
        Args:
            symbol: Stock symbol
            order_type: Order type (buy/sell)
            expected_price: Expected execution price
            fill_price: Actual fill price
            partial_fill: Whether fill was partial
        """
        if not self.track_fill_quality:
            return
        
        # Calculate slippage
        if order_type == "buy":
            slippage = (fill_price - expected_price) / expected_price
        else:
            slippage = (expected_price - fill_price) / expected_price
        
        slippage_bps = slippage * 10000  # Convert to basis points
        
        metric = FillMetric(
            symbol=symbol,
            order_type=order_type,
            expected_price=expected_price,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            partial_fill=partial_fill
        )
        self.fill_metrics.append(metric)
        
        # Update stats
        if symbol not in self.fill_stats:
            self.fill_stats[symbol] = {
                'count': 0,
                'total_slippage_bps': 0.0,
                'partial_fills': 0
            }
        
        stats = self.fill_stats[symbol]
        stats['count'] += 1
        stats['total_slippage_bps'] += abs(slippage_bps)
        if partial_fill:
            stats['partial_fills'] += 1
    
    def get_latency_summary(
        self,
        component: Optional[str] = None,
        minutes: int = 60
    ) -> Dict:
        """
        Get latency summary.
        
        Args:
            component: Optional component filter
            minutes: Time window in minutes
            
        Returns:
            Dictionary with latency statistics
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        # Filter recent metrics
        recent = [
            m for m in self.latency_metrics
            if m.timestamp >= cutoff
            and (component is None or m.component == component)
        ]
        
        if not recent:
            return {'count': 0}
        
        latencies = [m.latency_ms for m in recent]
        
        return {
            'count': len(recent),
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies)
        }
    
    def get_fill_quality_summary(
        self,
        symbol: Optional[str] = None,
        minutes: int = 60
    ) -> Dict:
        """
        Get fill quality summary.
        
        Args:
            symbol: Optional symbol filter
            minutes: Time window in minutes
            
        Returns:
            Dictionary with fill quality statistics
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        # Filter recent metrics
        recent = [
            m for m in self.fill_metrics
            if m.timestamp >= cutoff
            and (symbol is None or m.symbol == symbol)
        ]
        
        if not recent:
            return {'count': 0}
        
        slippages = [m.slippage_bps for m in recent]
        partial_count = sum(1 for m in recent if m.partial_fill)
        
        return {
            'count': len(recent),
            'avg_slippage_bps': np.mean(slippages),
            'median_slippage_bps': np.median(slippages),
            'max_slippage_bps': np.max(np.abs(slippages)),
            'partial_fill_rate': (partial_count / len(recent)) * 100 if recent else 0
        }
    
    def get_metrics_report(self) -> Dict:
        """
        Get comprehensive metrics report.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'latency': {
                'last_hour': self.get_latency_summary(minutes=60),
                'last_day': self.get_latency_summary(minutes=1440)
            },
            'fill_quality': {
                'last_hour': self.get_fill_quality_summary(minutes=60),
                'last_day': self.get_fill_quality_summary(minutes=1440)
            },
            'total_latency_records': len(self.latency_metrics),
            'total_fill_records': len(self.fill_metrics)
        }
    
    def get_component_latency(self, component: str) -> Dict:
        """
        Get latency stats for a specific component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary with component latency stats
        """
        stats = {}
        for key, data in self.latency_stats.items():
            if key.startswith(f"{component}."):
                operation = key.split('.', 1)[1]
                stats[operation] = {
                    'count': data['count'],
                    'avg_ms': data['sum'] / data['count'] if data['count'] > 0 else 0,
                    'min_ms': data['min'] if data['min'] != float('inf') else 0,
                    'max_ms': data['max']
                }
        return stats
    
    def reset_metrics(self) -> None:
        """Reset all metrics (for testing or periodic refresh)."""
        self.latency_metrics.clear()
        self.fill_metrics.clear()
        self.latency_stats.clear()
        self.fill_stats.clear()
        logger.info("Metrics reset")
