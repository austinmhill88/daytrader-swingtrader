"""
Unified launcher for trading system with GUI.
Starts both the trading bot and GUI interface.
"""
import sys
import argparse
import subprocess
import time
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import TradingBot
from gui.main_window import MainWindow, QApplication


def launch_ai_server():
    """Launch AI server in separate process."""
    logger.info("Starting AI server...")
    try:
        # Start AI server as subprocess
        process = subprocess.Popen(
            [sys.executable, "-m", "server.main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it time to start
        time.sleep(3)
        
        if process.poll() is None:
            logger.info("AI server started successfully")
            return process
        else:
            logger.warning("AI server failed to start")
            return None
    except Exception as e:
        logger.error(f"Failed to start AI server: {e}")
        return None


def main():
    """Main entry point for unified launcher."""
    parser = argparse.ArgumentParser(description="Trading System with GUI")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI (headless mode)'
    )
    parser.add_argument(
        '--gui-only',
        action='store_true',
        help='Run only GUI (trading system must be running separately)'
    )
    parser.add_argument(
        '--with-ai',
        action='store_true',
        help='Start AI server automatically'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode'
    )
    
    args = parser.parse_args()
    
    ai_server_process = None
    
    try:
        # Start AI server if requested
        if args.with_ai:
            ai_server_process = launch_ai_server()
        
        # GUI-only mode
        if args.gui_only:
            logger.info("Starting GUI in standalone mode")
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            window = MainWindow()
            window.show()
            return app.exec()
        
        # Headless mode (no GUI)
        if args.no_gui:
            logger.info("Starting trading bot in headless mode")
            bot = TradingBot(args.config)
            bot.start()
            return
        
        # Full mode: Trading bot + GUI
        logger.info("Starting trading system with GUI")
        
        # Create trading bot
        bot = TradingBot(args.config)
        
        # Create GUI application
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Create main window with trading system
        window = MainWindow(trading_system=bot)
        window.show()
        
        # Note: In production, you'd want to run the bot in a separate thread
        # For now, this is a basic integration showing the structure
        
        logger.info("=" * 80)
        logger.info("Trading System with GUI Started")
        logger.info("=" * 80)
        
        return app.exec()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup AI server
        if ai_server_process:
            logger.info("Stopping AI server...")
            ai_server_process.terminate()
            ai_server_process.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main() or 0)
