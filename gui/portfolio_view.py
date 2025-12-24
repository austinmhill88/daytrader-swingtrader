"""
Portfolio View - Equity curve, P&L tracking, and performance metrics.
"""
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from loguru import logger


class PortfolioView(QWidget):
    """
    Portfolio view showing equity curve and performance metrics.
    """
    
    def __init__(self, trading_system=None):
        """
        Initialize portfolio view.
        
        Args:
            trading_system: Optional trading system instance
        """
        super().__init__()
        self.trading_system = trading_system
        self.equity_history = []
        self.timestamps = []
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Summary metrics
        summary_group = self._create_summary_panel()
        layout.addWidget(summary_group)
        
        # Equity curve chart
        chart_group = QGroupBox("Equity Curve")
        chart_layout = QVBoxLayout(chart_group)
        
        # Create plot widget
        self.equity_plot = pg.PlotWidget()
        self.equity_plot.setBackground('w')
        self.equity_plot.setLabel('left', 'Equity ($)')
        self.equity_plot.setLabel('bottom', 'Time')
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot line
        self.equity_curve = self.equity_plot.plot(
            pen=pg.mkPen(color='b', width=2),
            name='Equity'
        )
        
        chart_layout.addWidget(self.equity_plot)
        layout.addWidget(chart_group)
        
        # Performance metrics
        perf_group = self._create_performance_panel()
        layout.addWidget(perf_group)
    
    def _create_summary_panel(self) -> QGroupBox:
        """Create summary metrics panel."""
        group = QGroupBox("Portfolio Summary")
        layout = QGridLayout(group)
        
        self.summary_metrics = {}
        metrics_config = [
            ("Starting Balance", "$0.00", 0, 0),
            ("Current Equity", "$0.00", 0, 1),
            ("Total P&L", "$0.00", 0, 2),
            ("Total Return", "0%", 0, 3),
        ]
        
        for name, default, row, col in metrics_config:
            label = QLabel(name + ":")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(label, row, col * 2)
            
            value_label = QLabel(default)
            value_font = QFont()
            value_font.setBold(True)
            value_font.setPointSize(12)
            value_label.setFont(value_font)
            layout.addWidget(value_label, row, col * 2 + 1)
            
            self.summary_metrics[name] = value_label
        
        return group
    
    def _create_performance_panel(self) -> QGroupBox:
        """Create performance metrics panel."""
        group = QGroupBox("Performance Metrics")
        layout = QGridLayout(group)
        
        self.perf_metrics = {}
        metrics_config = [
            ("Total Trades", "0", 0, 0),
            ("Winning Trades", "0", 0, 1),
            ("Losing Trades", "0", 0, 2),
            ("Win Rate", "0%", 0, 3),
            ("Largest Win", "$0.00", 1, 0),
            ("Largest Loss", "$0.00", 1, 1),
            ("Avg Win", "$0.00", 1, 2),
            ("Avg Loss", "$0.00", 1, 3),
            ("Profit Factor", "0.00", 2, 0),
            ("Sharpe Ratio", "0.00", 2, 1),
            ("Max Drawdown", "0%", 2, 2),
            ("Recovery Factor", "0.00", 2, 3),
        ]
        
        for name, default, row, col in metrics_config:
            label = QLabel(name + ":")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(label, row, col * 2)
            
            value_label = QLabel(default)
            value_font = QFont()
            value_font.setBold(True)
            value_label.setFont(value_font)
            layout.addWidget(value_label, row, col * 2 + 1)
            
            self.perf_metrics[name] = value_label
        
        return group
    
    def update_data(self):
        """Update portfolio view with latest data."""
        if not self.trading_system:
            return
        
        try:
            self._update_summary()
            self._update_equity_curve()
            self._update_performance()
        except Exception as e:
            logger.error(f"Error updating portfolio view: {e}")
    
    def _update_summary(self):
        """Update summary metrics."""
        if not hasattr(self.trading_system, 'portfolio'):
            return
        
        try:
            portfolio = self.trading_system.portfolio
            
            # Get current equity
            equity = portfolio.equity()
            
            # Update summary
            self.summary_metrics["Current Equity"].setText(f"${equity:,.2f}")
            
            # Calculate starting balance (would need to track this)
            # For now, assume we track it
            starting_balance = getattr(portfolio, 'starting_equity', equity)
            self.summary_metrics["Starting Balance"].setText(f"${starting_balance:,.2f}")
            
            # Total P&L
            total_pnl = equity - starting_balance
            total_return_pct = ((equity / starting_balance) - 1) * 100 if starting_balance > 0 else 0
            
            pnl_text = f"${total_pnl:,.2f}"
            return_text = f"{total_return_pct:.2f}%"
            
            # Color code
            if total_pnl > 0:
                self.summary_metrics["Total P&L"].setStyleSheet("color: green;")
                self.summary_metrics["Total Return"].setStyleSheet("color: green;")
            elif total_pnl < 0:
                self.summary_metrics["Total P&L"].setStyleSheet("color: red;")
                self.summary_metrics["Total Return"].setStyleSheet("color: red;")
            
            self.summary_metrics["Total P&L"].setText(pnl_text)
            self.summary_metrics["Total Return"].setText(return_text)
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
    
    def _update_equity_curve(self):
        """Update equity curve chart."""
        if not hasattr(self.trading_system, 'portfolio'):
            return
        
        try:
            portfolio = self.trading_system.portfolio
            equity = portfolio.equity()
            
            # Add current equity to history
            now = datetime.now()
            self.timestamps.append(now.timestamp())
            self.equity_history.append(equity)
            
            # Keep last 1000 points
            if len(self.equity_history) > 1000:
                self.timestamps = self.timestamps[-1000:]
                self.equity_history = self.equity_history[-1000:]
            
            # Update plot
            if len(self.equity_history) > 1:
                # Convert timestamps to relative seconds from start
                start_time = self.timestamps[0]
                x_data = [(t - start_time) for t in self.timestamps]
                
                self.equity_curve.setData(x=x_data, y=self.equity_history)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    def _update_performance(self):
        """Update performance metrics."""
        # These would be calculated from trade history
        # For now, just placeholder logic
        
        # In a real implementation, you would:
        # 1. Get trade history from trading system
        # 2. Calculate performance metrics
        # 3. Update labels
        
        pass
