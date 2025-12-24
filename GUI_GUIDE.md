# GUI User Guide

## Overview

The Trading GUI provides a real-time Windows desktop interface for monitoring and controlling your trading system. Built with PyQt6, it offers live updates, comprehensive metrics, and AI-powered assistance.

## Features

### 1. Trading Dashboard
- **Real-time Position Monitoring**: View all active positions with live P&L
- **Recent Signals**: Track latest trading signals from strategies
- **System Metrics**: Monitor equity, exposure, and performance

### 2. Portfolio View
- **Equity Curve**: Real-time chart of account equity
- **Performance Metrics**: Track wins, losses, drawdown, Sharpe ratio
- **Summary Statistics**: Overall portfolio health and returns

### 3. AI Assistant Console
- **Interactive Chat**: Ask questions about trading and strategies
- **Quick Actions**: Pre-configured prompts for common queries
- **Model Selection**: Choose between different AI models

### 4. Control Panel
- **Pause/Resume**: Temporarily halt trading
- **Emergency Stop**: Immediately flatten all positions and stop trading
- **System Status**: Real-time system health indicator

## Installation

### Prerequisites

```bash
pip install PyQt6 PyQt6-Charts matplotlib pyqtgraph
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Verify Installation

Test GUI components:
```python
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 OK')"
python -c "import pyqtgraph; print('pyqtgraph OK')"
```

## Launching the GUI

### Method 1: Unified Launcher (Recommended)

Start both trading system and GUI:
```bash
python launcher.py
```

With AI server:
```bash
python launcher.py --with-ai
```

### Method 2: GUI Only

If trading system is already running:
```bash
python launcher.py --gui-only
```

Or directly:
```bash
python -m gui.main_window
```

### Method 3: Integrated Mode

From Python:
```python
from gui.main_window import MainWindow, QApplication
from src.main import TradingBot

# Create trading bot
bot = TradingBot('config/config.yaml')

# Create GUI with trading system integration
app = QApplication(sys.argv)
window = MainWindow(trading_system=bot)
window.show()
app.exec()
```

## Using the GUI

### Trading Dashboard

#### Metrics Panel
- **Equity**: Current account value
- **Daily P&L**: Today's profit/loss (green = profit, red = loss)
- **Open Positions**: Number of active positions
- **Gross/Net Exposure**: Portfolio exposure percentages

#### Positions Table
- **Symbol**: Stock ticker
- **Side**: LONG (green) or SHORT (red)
- **Qty**: Number of shares
- **Entry**: Average entry price
- **Current**: Current market price
- **P&L**: Unrealized profit/loss with color coding
- **P&L%**: Percentage return on position

#### Signals Table
- **Time**: When signal was generated
- **Symbol**: Stock ticker
- **Strategy**: Which strategy generated the signal
- **Signal**: BUY/SELL/EXIT
- **Confidence**: Signal confidence score
- **Reason**: Why the signal was generated

### Portfolio View

#### Equity Curve
- Real-time chart showing account value over time
- Updated every second with new data point
- Zoom and pan with mouse

#### Summary Metrics
- **Starting Balance**: Initial account value
- **Current Equity**: Current account value
- **Total P&L**: Net profit/loss
- **Total Return**: Percentage return

#### Performance Metrics
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline

### AI Assistant Console

#### Basic Usage

1. **Type Question**: Enter your question in the input field
2. **Press Enter or Send**: Submit your query
3. **View Response**: AI response appears in chat window

Example questions:
```
"What should I monitor for risk management?"
"Explain the current market regime"
"Analyze my recent trading performance"
"What are good position sizing strategies?"
```

#### Quick Actions

Pre-configured prompts for common queries:
- **Market Summary**: Get overview of current conditions
- **Risk Check**: Risk management reminders
- **Strategy Tips**: Improve trading performance

#### Model Selection

Switch between different AI models:
- **qwen2.5:3b-instruct**: Fast, good for general queries
- **llama3.2:3b-instruct**: Alternative model

#### Status Indicator
- **Green (● Ready)**: AI server available
- **Orange (● Processing)**: Request in progress
- **Red (● Error)**: Error occurred

### Control Panel

#### Status Indicator
- **● LIVE** (Green): Trading active
- **● PAUSED** (Orange): Trading paused
- **● STOPPED** (Red): Emergency stop activated

#### Pause Trading
Temporarily halts all trading activity:
- Stops generating new signals
- Cancels pending orders
- Maintains existing positions
- Can be resumed

**When to use:**
- High market volatility
- News events
- System maintenance
- Testing changes

#### Resume Trading
Resumes normal trading operations:
- Re-enables signal generation
- Resumes order execution
- Requires confirmation

#### Emergency Stop
Immediately halts all trading and flattens positions:
- Cancels ALL open orders
- Closes ALL positions at market
- Disables all trading
- Requires manual restart

**When to use:**
- Critical system issues
- Unexpected market behavior
- Account protection
- Major news events

⚠️ **Warning**: This action cannot be undone!

## Configuration

### Update Refresh Rate

Edit `gui/main_window.py`:
```python
self.update_timer.start(1000)  # Update every 1000ms (1 second)
```

Adjust based on needs:
- 500ms: Very responsive, higher CPU
- 1000ms: Balanced (default)
- 2000ms: Less frequent, lower CPU

### Window Size

Edit `gui/main_window.py`:
```python
self.setGeometry(100, 100, 1400, 900)  # x, y, width, height
```

### Theme Customization

The GUI uses Qt's Fusion style. To customize:

```python
# In launcher.py or main_window.py
app.setStyle("Fusion")

# Optional: Dark theme
from PyQt6.QtGui import QPalette, QColor

palette = QPalette()
palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
app.setPalette(palette)
```

## Keyboard Shortcuts

Current shortcuts:
- **Enter**: Send message in AI console
- More shortcuts can be added as needed

To add custom shortcuts, edit relevant GUI files:
```python
from PyQt6.QtGui import QShortcut, QKeySequence

# Example: F5 to refresh
refresh_shortcut = QShortcut(QKeySequence("F5"), self)
refresh_shortcut.activated.connect(self._update_data)
```

## Performance Tips

### Reduce Update Frequency
If GUI feels sluggish:
1. Increase update timer interval (1000ms → 2000ms)
2. Reduce chart history points (1000 → 500)
3. Close unused tabs

### Optimize Charts
For better chart performance:
```python
# In portfolio_view.py
self.equity_plot.setClipToView(True)  # Only render visible area
self.equity_plot.setDownsampling(auto=True)  # Auto-downsample
```

### Monitor Resource Usage
Check CPU/RAM usage in Task Manager:
- GUI should use <5% CPU when idle
- ~50-100MB RAM typical

## Troubleshooting

### GUI Won't Start

**Error: "No module named 'PyQt6'"**
```bash
pip install PyQt6 PyQt6-Charts
```

**Error: "No module named 'pyqtgraph'"**
```bash
pip install pyqtgraph
```

### No Data Showing

1. **Check Trading System**: Ensure trading system is running
2. **Verify Connection**: Check system is passed to MainWindow
3. **Check Logs**: Look for errors in `logs/runtime.log`

### AI Console Not Working

1. **Check AI Server**: Verify server is running at http://127.0.0.1:8000
2. **Test Endpoint**: `curl http://127.0.0.1:8000/health`
3. **Check Config**: Verify `ai_server.enabled: true` in config.yaml
4. **View Logs**: Check `logs/ai_tools.log`

### Freezing or Slow Response

1. **AI Requests**: AI console uses background thread (should not freeze)
2. **Update Rate**: Increase timer interval if too frequent
3. **Chart Data**: Limit history points in equity curve
4. **System Resources**: Close other applications

### Chart Not Updating

1. **Check Data Source**: Verify trading system is providing data
2. **Verify Update Timer**: Ensure timer is running
3. **Look for Errors**: Check console output for exceptions

### Controls Not Working

1. **Trading System**: Ensure system has `admin_controls` attribute
2. **Permissions**: Verify user has permission to control system
3. **State**: Check system is in correct state for operation

## Advanced Usage

### Running Multiple Instances

To monitor multiple accounts:
1. Start each trading system with different config
2. Launch separate GUI for each:
```bash
python -m gui.main_window --config config/account1.yaml
python -m gui.main_window --config config/account2.yaml
```

### Custom Widgets

Add custom panels by editing GUI files:

```python
# In main_window.py
from .custom_panel import CustomPanel

self.custom_panel = CustomPanel(self.trading_system)
self.tabs.addTab(self.custom_panel, "Custom View")
```

### Data Export

Export data from GUI:
```python
# Add to portfolio_view.py
def export_equity_curve(self):
    import pandas as pd
    df = pd.DataFrame({
        'timestamp': self.timestamps,
        'equity': self.equity_history
    })
    df.to_csv('equity_export.csv', index=False)
```

## Best Practices

1. **Monitor Regularly**: Check GUI at market open, mid-day, and close
2. **Use Pause**: Pause trading during volatile periods, don't emergency stop
3. **Emergency Stop**: Only use in true emergencies
4. **AI Assistant**: Use for analysis and learning, not real-time decisions
5. **Update Frequency**: Balance between responsiveness and performance
6. **Window Size**: Adjust for your monitor resolution
7. **Take Breaks**: Don't watch every tick - trust your system

## Screenshots

(Screenshots would be added here after GUI is running)

Key views to capture:
1. Main dashboard with active positions
2. Portfolio equity curve
3. AI console interaction
4. Control panel with status indicators

## Future Enhancements

Planned features:
- [ ] Order entry panel for manual trades
- [ ] Strategy configuration panel
- [ ] Advanced charting with indicators
- [ ] Alert notifications
- [ ] Trade journal integration
- [ ] Performance analytics dashboard
- [ ] Risk heat map
- [ ] Real-time news feed

## Support

For GUI issues:
1. Check console output for errors
2. Review logs: `logs/runtime.log`
3. Verify PyQt6 installation
4. Test individual components
5. Report bugs with screenshots

## References

- PyQt6 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- pyqtgraph Documentation: https://pyqtgraph.readthedocs.io/
- Qt Designer: Use for visual GUI design (optional)
