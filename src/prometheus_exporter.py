"""
Prometheus metrics exporter for trading system monitoring.
"""
from prometheus_client import Gauge, Counter, Histogram, start_http_server
from loguru import logger
from typing import Dict, Optional


class PrometheusExporter:
    """
    Exposes trading system metrics via Prometheus.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Prometheus exporter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        metrics_config = config.get('metrics', {})
        
        self.enabled = metrics_config.get('enable_prometheus', False)
        self.port = metrics_config.get('port', 9101)
        
        if not self.enabled:
            logger.info("Prometheus exporter disabled")
            return
        
        # Define metrics
        
        # Portfolio metrics
        self.equity_gauge = Gauge(
            'trading_equity_usd',
            'Current portfolio equity in USD'
        )
        
        self.cash_gauge = Gauge(
            'trading_cash_usd',
            'Current cash balance in USD'
        )
        
        self.positions_value_gauge = Gauge(
            'trading_positions_value_usd',
            'Total value of positions in USD'
        )
        
        self.daily_pnl_gauge = Gauge(
            'trading_daily_pnl_usd',
            'Daily profit/loss in USD'
        )
        
        self.daily_pnl_pct_gauge = Gauge(
            'trading_daily_pnl_pct',
            'Daily profit/loss percentage'
        )
        
        # Exposure metrics
        self.gross_exposure_gauge = Gauge(
            'trading_gross_exposure_pct',
            'Gross exposure percentage'
        )
        
        self.net_exposure_gauge = Gauge(
            'trading_net_exposure_pct',
            'Net exposure percentage'
        )
        
        self.long_exposure_gauge = Gauge(
            'trading_long_exposure_usd',
            'Long exposure in USD'
        )
        
        self.short_exposure_gauge = Gauge(
            'trading_short_exposure_usd',
            'Short exposure in USD'
        )
        
        # Position metrics
        self.num_positions_gauge = Gauge(
            'trading_num_positions',
            'Number of open positions'
        )
        
        self.num_long_positions_gauge = Gauge(
            'trading_num_long_positions',
            'Number of long positions'
        )
        
        self.num_short_positions_gauge = Gauge(
            'trading_num_short_positions',
            'Number of short positions'
        )
        
        # Order metrics
        self.orders_placed_counter = Counter(
            'trading_orders_placed_total',
            'Total number of orders placed',
            ['strategy', 'side']
        )
        
        self.orders_filled_counter = Counter(
            'trading_orders_filled_total',
            'Total number of orders filled',
            ['strategy', 'side']
        )
        
        self.orders_rejected_counter = Counter(
            'trading_orders_rejected_total',
            'Total number of orders rejected',
            ['reason']
        )
        
        self.orders_cancelled_counter = Counter(
            'trading_orders_cancelled_total',
            'Total number of orders cancelled'
        )
        
        # Latency metrics
        self.order_latency_histogram = Histogram(
            'trading_order_latency_ms',
            'Order placement latency in milliseconds',
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        )
        
        self.data_latency_histogram = Histogram(
            'trading_data_latency_ms',
            'Data feed latency in milliseconds',
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        )
        
        # Risk metrics
        self.kill_switch_active_gauge = Gauge(
            'trading_kill_switch_active',
            'Kill switch status (1=active, 0=inactive)'
        )
        
        self.daily_drawdown_gauge = Gauge(
            'trading_daily_drawdown_pct',
            'Current daily drawdown percentage'
        )
        
        self.max_drawdown_gauge = Gauge(
            'trading_max_drawdown_pct',
            'Maximum drawdown percentage'
        )
        
        # Trade metrics
        self.trades_today_counter = Counter(
            'trading_trades_today_total',
            'Total trades today',
            ['strategy']
        )
        
        self.winning_trades_counter = Counter(
            'trading_winning_trades_total',
            'Total winning trades',
            ['strategy']
        )
        
        self.losing_trades_counter = Counter(
            'trading_losing_trades_total',
            'Total losing trades',
            ['strategy']
        )
        
        # Slippage metrics
        self.slippage_histogram = Histogram(
            'trading_slippage_bps',
            'Slippage in basis points',
            buckets=[1, 2, 5, 10, 15, 20, 30, 50, 100]
        )
        
        # System metrics
        self.api_errors_counter = Counter(
            'trading_api_errors_total',
            'Total API errors',
            ['component']
        )
        
        self.reconnections_counter = Counter(
            'trading_reconnections_total',
            'Total reconnection attempts',
            ['component']
        )
        
        logger.info(f"Prometheus exporter initialized on port {self.port}")
    
    def start(self) -> None:
        """Start the Prometheus HTTP server."""
        if not self.enabled:
            return
        
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def update_portfolio_metrics(
        self,
        equity: float,
        cash: float,
        positions_value: float,
        daily_pnl: float,
        daily_pnl_pct: float
    ) -> None:
        """
        Update portfolio metrics.
        
        Args:
            equity: Current equity
            cash: Current cash
            positions_value: Total positions value
            daily_pnl: Daily P&L in USD
            daily_pnl_pct: Daily P&L percentage
        """
        if not self.enabled:
            return
        
        self.equity_gauge.set(equity)
        self.cash_gauge.set(cash)
        self.positions_value_gauge.set(positions_value)
        self.daily_pnl_gauge.set(daily_pnl)
        self.daily_pnl_pct_gauge.set(daily_pnl_pct)
    
    def update_exposure_metrics(
        self,
        gross_exposure_pct: float,
        net_exposure_pct: float,
        long_exposure: float,
        short_exposure: float
    ) -> None:
        """
        Update exposure metrics.
        
        Args:
            gross_exposure_pct: Gross exposure percentage
            net_exposure_pct: Net exposure percentage
            long_exposure: Long exposure in USD
            short_exposure: Short exposure in USD
        """
        if not self.enabled:
            return
        
        self.gross_exposure_gauge.set(gross_exposure_pct)
        self.net_exposure_gauge.set(net_exposure_pct)
        self.long_exposure_gauge.set(long_exposure)
        self.short_exposure_gauge.set(short_exposure)
    
    def update_position_metrics(
        self,
        num_positions: int,
        num_long: int,
        num_short: int
    ) -> None:
        """
        Update position count metrics.
        
        Args:
            num_positions: Total number of positions
            num_long: Number of long positions
            num_short: Number of short positions
        """
        if not self.enabled:
            return
        
        self.num_positions_gauge.set(num_positions)
        self.num_long_positions_gauge.set(num_long)
        self.num_short_positions_gauge.set(num_short)
    
    def record_order_placed(self, strategy: str, side: str) -> None:
        """Record an order placement."""
        if not self.enabled:
            return
        self.orders_placed_counter.labels(strategy=strategy, side=side).inc()
    
    def record_order_filled(self, strategy: str, side: str) -> None:
        """Record an order fill."""
        if not self.enabled:
            return
        self.orders_filled_counter.labels(strategy=strategy, side=side).inc()
    
    def record_order_rejected(self, reason: str) -> None:
        """Record an order rejection."""
        if not self.enabled:
            return
        self.orders_rejected_counter.labels(reason=reason).inc()
    
    def record_order_cancelled(self) -> None:
        """Record an order cancellation."""
        if not self.enabled:
            return
        self.orders_cancelled_counter.inc()
    
    def record_order_latency(self, latency_ms: float) -> None:
        """Record order placement latency."""
        if not self.enabled:
            return
        self.order_latency_histogram.observe(latency_ms)
    
    def record_data_latency(self, latency_ms: float) -> None:
        """Record data feed latency."""
        if not self.enabled:
            return
        self.data_latency_histogram.observe(latency_ms)
    
    def update_risk_metrics(
        self,
        kill_switch_active: bool,
        daily_drawdown_pct: float,
        max_drawdown_pct: float
    ) -> None:
        """
        Update risk metrics.
        
        Args:
            kill_switch_active: Whether kill switch is active
            daily_drawdown_pct: Current daily drawdown
            max_drawdown_pct: Maximum drawdown
        """
        if not self.enabled:
            return
        
        self.kill_switch_active_gauge.set(1 if kill_switch_active else 0)
        self.daily_drawdown_gauge.set(daily_drawdown_pct)
        self.max_drawdown_gauge.set(max_drawdown_pct)
    
    def record_trade(self, strategy: str, is_winning: bool) -> None:
        """
        Record a completed trade.
        
        Args:
            strategy: Strategy name
            is_winning: Whether trade was profitable
        """
        if not self.enabled:
            return
        
        self.trades_today_counter.labels(strategy=strategy).inc()
        if is_winning:
            self.winning_trades_counter.labels(strategy=strategy).inc()
        else:
            self.losing_trades_counter.labels(strategy=strategy).inc()
    
    def record_slippage(self, slippage_bps: float) -> None:
        """Record slippage in basis points."""
        if not self.enabled:
            return
        self.slippage_histogram.observe(abs(slippage_bps))
    
    def record_api_error(self, component: str) -> None:
        """Record an API error."""
        if not self.enabled:
            return
        self.api_errors_counter.labels(component=component).inc()
    
    def record_reconnection(self, component: str) -> None:
        """Record a reconnection attempt."""
        if not self.enabled:
            return
        self.reconnections_counter.labels(component=component).inc()
