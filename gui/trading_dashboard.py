"""
Trading Dashboard - Real-time view of active trades and system status.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from loguru import logger


class TradingDashboard(QWidget):
    """
    Dashboard showing active trades, signals, and system metrics.
    """
    
    def __init__(self, trading_system=None):
        """
        Initialize trading dashboard.
        
        Args:
            trading_system: Optional trading system instance
        """
        super().__init__()
        self.trading_system = trading_system
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Metrics panel
        metrics_group = self._create_metrics_panel()
        layout.addWidget(metrics_group)
        
        # Active positions table
        positions_group = QGroupBox("Active Positions")
        positions_layout = QVBoxLayout(positions_group)
        
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(9)
        self.positions_table.setHorizontalHeaderLabels([
            "Symbol", "Side", "Qty", "Entry", "Current", "P&L", "P&L%", "Strategy", "Time"
        ])
        self.positions_table.setAlternatingRowColors(True)
        positions_layout.addWidget(self.positions_table)
        
        layout.addWidget(positions_group)
        
        # Recent signals table
        signals_group = QGroupBox("Recent Signals")
        signals_layout = QVBoxLayout(signals_group)
        
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(6)
        self.signals_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Strategy", "Signal", "Confidence", "Reason"
        ])
        self.signals_table.setAlternatingRowColors(True)
        self.signals_table.setMaximumHeight(200)
        signals_layout.addWidget(self.signals_table)
        
        layout.addWidget(signals_group)
    
    def _create_metrics_panel(self) -> QGroupBox:
        """Create metrics display panel."""
        group = QGroupBox("System Metrics")
        layout = QGridLayout(group)
        
        # Create metric labels
        self.metrics = {}
        metrics_config = [
            ("Equity", "$0.00", 0, 0),
            ("Daily P&L", "$0.00", 0, 1),
            ("Daily P&L %", "0.00%", 0, 2),
            ("Open Positions", "0", 0, 3),
            ("Today's Trades", "0", 1, 0),
            ("Win Rate", "0%", 1, 1),
            ("Gross Exposure", "0%", 1, 2),
            ("Net Exposure", "0%", 1, 3),
        ]
        
        for name, default, row, col in metrics_config:
            # Label
            label = QLabel(name + ":")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(label, row, col * 2)
            
            # Value
            value_label = QLabel(default)
            value_font = QFont()
            value_font.setBold(True)
            value_font.setPointSize(11)
            value_label.setFont(value_font)
            layout.addWidget(value_label, row, col * 2 + 1)
            
            self.metrics[name] = value_label
        
        return group
    
    def update_data(self):
        """Update dashboard with latest data."""
        if not self.trading_system:
            return
        
        try:
            # Update metrics
            self._update_metrics()
            
            # Update positions table
            self._update_positions()
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def _update_metrics(self):
        """Update metric labels."""
        if not hasattr(self.trading_system, 'portfolio'):
            return
        
        try:
            portfolio = self.trading_system.portfolio
            
            # Equity
            equity = portfolio.equity()
            self.metrics["Equity"].setText(f"${equity:,.2f}")
            
            # Daily P&L
            daily_pnl = portfolio.daily_pnl()
            daily_pnl_pct = portfolio.daily_pnl_pct()
            
            pnl_text = f"${daily_pnl:,.2f}"
            pnl_pct_text = f"{daily_pnl_pct:.2f}%"
            
            # Color code P&L
            if daily_pnl > 0:
                self.metrics["Daily P&L"].setStyleSheet("color: green;")
                self.metrics["Daily P&L %"].setStyleSheet("color: green;")
            elif daily_pnl < 0:
                self.metrics["Daily P&L"].setStyleSheet("color: red;")
                self.metrics["Daily P&L %"].setStyleSheet("color: red;")
            else:
                self.metrics["Daily P&L"].setStyleSheet("")
                self.metrics["Daily P&L %"].setStyleSheet("")
            
            self.metrics["Daily P&L"].setText(pnl_text)
            self.metrics["Daily P&L %"].setText(pnl_pct_text)
            
            # Positions
            positions = portfolio.positions()
            self.metrics["Open Positions"].setText(str(len(positions)))
            
            # Exposure
            exposure = portfolio.calculate_exposure()
            self.metrics["Gross Exposure"].setText(f"{exposure['gross_pct']:.1f}%")
            self.metrics["Net Exposure"].setText(f"{exposure['net_pct']:.1f}%")
            
            # Today's trades (would need to track this)
            # self.metrics["Today's Trades"].setText("0")
            # self.metrics["Win Rate"].setText("0%")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _update_positions(self):
        """Update positions table."""
        if not hasattr(self.trading_system, 'portfolio'):
            return
        
        try:
            portfolio = self.trading_system.portfolio
            positions = portfolio.positions()
            
            self.positions_table.setRowCount(len(positions))
            
            for i, pos in enumerate(positions):
                # Symbol
                self.positions_table.setItem(i, 0, QTableWidgetItem(pos.symbol))
                
                # Side
                side_item = QTableWidgetItem(pos.side.upper())
                if pos.side == 'long':
                    side_item.setForeground(QColor(0, 150, 0))
                else:
                    side_item.setForeground(QColor(200, 0, 0))
                self.positions_table.setItem(i, 1, side_item)
                
                # Quantity
                self.positions_table.setItem(i, 2, QTableWidgetItem(str(pos.qty)))
                
                # Entry price
                self.positions_table.setItem(i, 3, QTableWidgetItem(f"${pos.avg_entry_price:.2f}"))
                
                # Current price
                self.positions_table.setItem(i, 4, QTableWidgetItem(f"${pos.current_price:.2f}"))
                
                # P&L
                pnl = pos.unrealized_pl
                pnl_pct = pos.unrealized_plpc
                
                pnl_item = QTableWidgetItem(f"${pnl:.2f}")
                pnl_pct_item = QTableWidgetItem(f"{pnl_pct:.2f}%")
                
                if pnl > 0:
                    pnl_item.setForeground(QColor(0, 150, 0))
                    pnl_pct_item.setForeground(QColor(0, 150, 0))
                elif pnl < 0:
                    pnl_item.setForeground(QColor(200, 0, 0))
                    pnl_pct_item.setForeground(QColor(200, 0, 0))
                
                self.positions_table.setItem(i, 5, pnl_item)
                self.positions_table.setItem(i, 6, pnl_pct_item)
                
                # Strategy (if available)
                self.positions_table.setItem(i, 7, QTableWidgetItem(""))
                
                # Time (if available)
                self.positions_table.setItem(i, 8, QTableWidgetItem(""))
            
            # Resize columns to contents
            self.positions_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error updating positions table: {e}")
