"""
Main window for trading GUI application.
"""
import sys
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QStatusBar, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from loguru import logger

from .trading_dashboard import TradingDashboard
from .portfolio_view import PortfolioView
from .ai_console import AIConsole


class MainWindow(QMainWindow):
    """
    Main GUI window for trading system.
    Provides real-time monitoring and control interface.
    """
    
    def __init__(self, trading_system=None):
        """
        Initialize main window.
        
        Args:
            trading_system: Optional trading system instance for integration
        """
        super().__init__()
        self.trading_system = trading_system
        self.setWindowTitle("DayTrader SwingTrader - Trading System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Setup UI
        self._setup_ui()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(1000)  # Update every second
        
        logger.info("Main GUI window initialized")
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Header with controls
        header = self._create_header()
        layout.addWidget(header)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        
        # Dashboard tab
        self.dashboard = TradingDashboard(self.trading_system)
        self.tabs.addTab(self.dashboard, "📊 Trading Dashboard")
        
        # Portfolio tab
        self.portfolio_view = PortfolioView(self.trading_system)
        self.tabs.addTab(self.portfolio_view, "💼 Portfolio & P&L")
        
        # AI Console tab
        self.ai_console = AIConsole()
        self.tabs.addTab(self.ai_console, "🤖 AI Assistant")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System Ready")
    
    def _create_header(self) -> QWidget:
        """Create header with system controls."""
        header = QWidget()
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("Trading System Control Panel")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("● LIVE")
        self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.status_label)
        
        # Control buttons
        self.pause_btn = QPushButton("⏸ Pause Trading")
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        layout.addWidget(self.pause_btn)
        
        self.resume_btn = QPushButton("▶ Resume Trading")
        self.resume_btn.clicked.connect(self._on_resume_clicked)
        self.resume_btn.setEnabled(False)
        layout.addWidget(self.resume_btn)
        
        self.emergency_btn = QPushButton("🛑 Emergency Stop")
        self.emergency_btn.setStyleSheet("background-color: #ff4444; color: white; font-weight: bold;")
        self.emergency_btn.clicked.connect(self._on_emergency_clicked)
        layout.addWidget(self.emergency_btn)
        
        return header
    
    def _update_data(self):
        """Update all views with latest data."""
        try:
            # Update dashboard
            self.dashboard.update_data()
            
            # Update portfolio view
            self.portfolio_view.update_data()
            
            # Update status
            if self.trading_system:
                # Check if trading is paused
                try:
                    if hasattr(self.trading_system, 'admin_controls'):
                        if self.trading_system.admin_controls.is_paused():
                            self.status_label.setText("● PAUSED")
                            self.status_label.setStyleSheet("color: orange; font-weight: bold; font-size: 14px;")
                        else:
                            self.status_label.setText("● LIVE")
                            self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error updating GUI data: {e}")
    
    def _on_pause_clicked(self):
        """Handle pause button click."""
        if self.trading_system and hasattr(self.trading_system, 'admin_controls'):
            try:
                self.trading_system.admin_controls.pause_trading(reason="User requested via GUI")
                self.pause_btn.setEnabled(False)
                self.resume_btn.setEnabled(True)
                self.status_bar.showMessage("Trading paused")
                logger.info("Trading paused via GUI")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to pause trading: {e}")
        else:
            self.status_bar.showMessage("Trading system not connected")
    
    def _on_resume_clicked(self):
        """Handle resume button click."""
        if self.trading_system and hasattr(self.trading_system, 'admin_controls'):
            try:
                self.trading_system.admin_controls.resume_trading(confirm=True)
                self.pause_btn.setEnabled(True)
                self.resume_btn.setEnabled(False)
                self.status_bar.showMessage("Trading resumed")
                logger.info("Trading resumed via GUI")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to resume trading: {e}")
        else:
            self.status_bar.showMessage("Trading system not connected")
    
    def _on_emergency_clicked(self):
        """Handle emergency stop button click."""
        reply = QMessageBox.question(
            self, 
            "Emergency Stop",
            "Are you sure you want to trigger an emergency stop?\n\n"
            "This will:\n"
            "- Cancel all open orders\n"
            "- Flatten all positions\n"
            "- Halt all trading\n",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.trading_system and hasattr(self.trading_system, 'admin_controls'):
                try:
                    self.trading_system.admin_controls.emergency_halt(reason="User triggered via GUI")
                    self.status_label.setText("● STOPPED")
                    self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
                    self.status_bar.showMessage("EMERGENCY STOP ACTIVATED")
                    logger.warning("Emergency stop triggered via GUI")
                    
                    QMessageBox.information(
                        self, 
                        "Emergency Stop",
                        "Emergency stop activated successfully.\n\n"
                        "All positions have been flattened and trading is halted."
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to trigger emergency stop: {e}")
            else:
                self.status_bar.showMessage("Trading system not connected")
    
    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?\n\n"
            "This will close the GUI but the trading system will continue running.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("GUI application closing")
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point for GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
