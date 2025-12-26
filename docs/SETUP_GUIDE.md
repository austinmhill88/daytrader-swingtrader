# CI/CD and Monitoring Setup Guide

This guide explains how to set up and use the CI/CD workflows, monitoring infrastructure, and promotion gates that have been added to the trading system.

## Table of Contents

1. [CI/CD Workflows Setup](#cicd-workflows-setup)
2. [Monitoring Setup (Prometheus & Grafana)](#monitoring-setup)
3. [Model Promotion Gates](#model-promotion-gates)
4. [Alert Configuration](#alert-configuration)
5. [Hardware Optimization](#hardware-optimization)

---

## CI/CD Workflows Setup

### Prerequisites

- GitHub repository with Actions enabled
- (Optional) Alpaca API credentials for nightly backtests
- (Optional) Slack webhook URL for notifications

### GitHub Secrets Configuration

Set up the following secrets in your GitHub repository (Settings → Secrets and variables → Actions):

```
APCA_API_KEY_ID=your_alpaca_key_id
APCA_API_SECRET_KEY=your_alpaca_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Workflows

#### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers**:
- Push to `main`, `develop`, or `copilot/**` branches
- Pull requests to `main` or `develop`

**What it does**:
- Lints code with `ruff`
- Type checks with `mypy`
- Runs unit tests (if present)
- Validates YAML configuration files
- Checks for hardcoded secrets
- Uploads test coverage to Codecov

**Manual run**:
```bash
# From GitHub UI: Actions → CI → Run workflow
```

#### 2. Nightly Backtest Workflow (`.github/workflows/nightly-backtest.yml`)

**Triggers**:
- Scheduled daily at 2:00 AM ET (6:00 AM UTC)
- Manual trigger via GitHub Actions UI

**What it does**:
- Fetches latest market data
- Refreshes universe
- Runs walk-forward backtest
- Evaluates promotion gates
- Generates performance report
- Notifies on Slack
- Creates GitHub issue if gates fail

**Manual run**:
```bash
# From GitHub UI: Actions → Nightly Backtest → Run workflow
# Options:
#   - Retrain models: true/false
#   - Symbols to test: comma-separated list (e.g., AAPL,MSFT,GOOGL)
```

**Viewing results**:
- Go to Actions → Nightly Backtest → Select run
- Download artifacts: `nightly-backtest-results-{run_number}`
- Contains:
  - `backtest_results_YYYYMMDD.json` - Raw metrics
  - `nightly_report_YYYYMMDD.txt` - Human-readable summary
  - `gate_summary.json` - Promotion gate evaluation

#### 3. Docker Build Workflow (`.github/workflows/docker-build.yml`)

**Triggers**:
- Push to `main` branch
- Git tags matching `v*.*.*` pattern
- Pull requests (builds but doesn't push)

**What it does**:
- Builds Docker image with Buildx
- Pushes to GitHub Container Registry (ghcr.io)
- Runs Trivy vulnerability scanner
- Tests the image
- Generates deployment instructions

**Using the image**:
```bash
# Pull the latest image
docker pull ghcr.io/austinmhill88/daytrader-swingtrader:latest

# Run the container
docker run -d --name trading-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  ghcr.io/austinmhill88/daytrader-swingtrader:latest
```

---

## Monitoring Setup

### Prerequisites

- Docker and Docker Compose (recommended)
- OR manual installation of Prometheus and Grafana

### Option 1: Docker Compose (Recommended)

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin  # Change this!
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

Start monitoring stack:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### Option 2: Manual Installation

#### Install Prometheus

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.linux-amd64.tar.gz
tar -xvf prometheus-2.48.0.linux-amd64.tar.gz
cd prometheus-2.48.0.linux-amd64

# Copy config
cp /path/to/repo/monitoring/prometheus.yml .

# Run Prometheus
./prometheus --config.file=prometheus.yml
```

Access at: http://localhost:9090

#### Install Grafana

```bash
# Ubuntu/Debian
sudo apt-get install -y adduser libfontconfig1
wget https://dl.grafana.com/oss/release/grafana_10.2.2_amd64.deb
sudo dpkg -i grafana_10.2.2_amd64.deb

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Access at: http://localhost:3000 (default credentials: admin/admin)

### Configure Grafana

1. **Add Prometheus as a data source**:
   - Go to Configuration → Data Sources → Add data source
   - Select Prometheus
   - URL: `http://localhost:9090` (or `http://prometheus:9090` if using Docker)
   - Save & Test

2. **Import dashboards**:
   - Go to Dashboards → Import
   - Upload `grafana/dashboards/trading-overview.json`
   - Upload `grafana/dashboards/strategy-performance.json`
   - Select Prometheus as the data source

3. **Set up alerts** (optional):
   - Go to Alerting → Notification channels
   - Add Slack, email, or other channels
   - Configure alert rules based on metrics

### Implement Metrics Exporter in Trading System

Add Prometheus metrics exporter to your trading system code:

```python
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Define metrics
pnl_gauge = Gauge('trading_pnl_total', 'Total P&L', ['type'])
positions_gauge = Gauge('trading_positions_open', 'Open positions', ['strategy'])
exposure_gauge = Gauge('trading_exposure_pct', 'Exposure percentage', ['type'])
kill_switch_gauge = Gauge('trading_kill_switch', 'Kill switch status (0=OK, 1=ACTIVE)')
order_latency = Histogram('trading_order_latency_seconds', 'Order placement latency', ['percentile'])
fill_rate_gauge = Gauge('trading_fill_rate', 'Fill rate percentage', ['strategy'])
slippage_gauge = Gauge('trading_slippage_bps', 'Slippage in basis points', ['strategy'])
errors_counter = Counter('trading_errors_total', 'Total errors', ['component', 'severity'])

# Start metrics server on port 9101
start_http_server(9101)

# Update metrics in your trading loop
pnl_gauge.labels(type='realized').set(realized_pnl)
pnl_gauge.labels(type='unrealized').set(unrealized_pnl)
positions_gauge.labels(strategy='intraday_mean_reversion').set(num_positions)
exposure_gauge.labels(type='gross').set(gross_exposure_pct)
kill_switch_gauge.set(1 if kill_switch_active else 0)
```

---

## Model Promotion Gates

### Configuration

Edit `config/promotion_gates.yaml` to customize thresholds:

```yaml
performance_thresholds:
  min_sortino_ratio: 1.0
  min_sharpe_ratio: 1.0
  max_drawdown_pct: 10.0
  min_win_rate: 0.45
  min_total_trades: 100

anti_overfitting:
  max_pbo: 0.5
  min_deflated_sharpe: 0.5

stability:
  max_shap_cv: 0.3
  min_positive_windows_pct: 0.70
```

### Usage in Training Pipeline

```python
import yaml
from src.ml_trainer import MLTrainer

# Load promotion gates
with open('config/promotion_gates.yaml', 'r') as f:
    gates = yaml.safe_load(f)

# Train model
trainer = MLTrainer(config)
model, metrics = trainer.train_model(strategy='intraday_mean_reversion')

# Evaluate gates
from src.promotion_gates import evaluate_gates
gate_results = evaluate_gates(metrics, gates)

if gate_results['passed']:
    # Promote model to production
    trainer.promote_model(model.id)
    print(f"✅ Model {model.id} promoted")
else:
    print(f"⚠️ Model failed gates: {gate_results['failed_gates']}")
```

### Anti-Overfitting Diagnostics

Implement these in your training pipeline:

```python
from src.validation import (
    compute_probability_of_backtest_overfitting,
    compute_deflated_sharpe_ratio,
    compute_whites_reality_check,
    compute_shap_stability
)

# After walk-forward validation
pbo = compute_probability_of_backtest_overfitting(oos_returns)
dsr = compute_deflated_sharpe_ratio(sharpe_ratio, num_trials, num_samples)
wrc_pvalue = compute_whites_reality_check(strategy_returns, benchmark_returns)
shap_cv = compute_shap_stability(model, X_folds, top_n=10)

print(f"PBO: {pbo:.3f} (threshold: {gates['anti_overfitting']['max_pbo']})")
print(f"DSR: {dsr:.3f} (threshold: {gates['anti_overfitting']['min_deflated_sharpe']})")
print(f"SHAP CV: {shap_cv:.3f} (threshold: {gates['stability']['max_shap_cv']})")
```

---

## Alert Configuration

### Customize Alerts

Edit `config/alerts.yaml`:

```yaml
channels:
  slack:
    enabled: true
    webhook_url: ${SLACK_WEBHOOK_URL}
  
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    sender_email: ${ALERT_EMAIL_SENDER}
    recipient_emails:
      - your_email@example.com

risk:
  approaching_drawdown:
    threshold_pct: 1.5
    message: "⚠️ Daily drawdown approaching limit: {drawdown_pct:.2f}%"
```

### Implement Alert System

```python
import yaml
import requests

class AlertManager:
    def __init__(self, config_path='config/alerts.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def send_alert(self, alert_type, severity, **kwargs):
        alert_config = self._get_alert_config(alert_type)
        if not alert_config['enabled']:
            return
        
        message = alert_config['message'].format(**kwargs)
        
        # Send to configured channels
        if self.config['channels']['slack']['enabled']:
            self._send_slack(message, severity)
        
        if self.config['channels']['email']['enabled']:
            self._send_email(message, severity)
    
    def _send_slack(self, message, severity):
        webhook_url = self.config['channels']['slack']['webhook_url']
        color = self.config['severity_levels'][severity]['color']
        
        payload = {
            "attachments": [{
                "color": color,
                "text": message
            }]
        }
        requests.post(webhook_url, json=payload)

# Usage
alert_manager = AlertManager()
alert_manager.send_alert(
    'approaching_drawdown',
    severity='warning',
    drawdown_pct=1.6
)
```

---

## Hardware Optimization

### For AMD Ryzen 9 9950x + RTX 5070

#### CPU Pinning

Pin data processing to specific cores:

```python
import os
import psutil

# Reserve cores 0-15 for trading, 16-31 for data processing
def set_cpu_affinity(process_type='trading'):
    if process_type == 'trading':
        cores = list(range(0, 16))
    elif process_type == 'data_processing':
        cores = list(range(16, 32))
    
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cores)
```

#### GPU Acceleration

Enable GPU for LightGBM/XGBoost:

```python
import lightgbm as lgb

# Train with GPU
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05
}

model = lgb.train(params, train_data, num_boost_round=500)
```

For CUDA setup:
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify GPU is available
nvidia-smi

# Install GPU-enabled libraries
pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
pip install xgboost[gpu]
```

#### Memory Optimization

Configure for 64GB DDR5:

```python
import os

# Set memory limits for parallel processing
os.environ['OMP_NUM_THREADS'] = '16'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '16'  # Intel MKL threads

# LightGBM memory budget
params['max_bin'] = 255  # Lower memory usage
params['num_threads'] = 16
```

#### NVMe SSD Optimization

Ensure data is on NVMe partition:

```bash
# Check mount points
df -h

# Ensure data directory is on NVMe
# /data should be on /dev/nvme0n1 (Samsung 990 Pro)
```

---

## Next Steps

1. **Review and customize configurations**:
   - Edit `config/promotion_gates.yaml` for your risk tolerance
   - Edit `config/alerts.yaml` for your notification preferences

2. **Set up monitoring**:
   - Deploy Prometheus and Grafana
   - Implement metrics exporter in trading system

3. **Configure CI/CD**:
   - Add GitHub secrets
   - Test workflows manually

4. **Calibrate thresholds**:
   - Run several backtests
   - Observe metric distributions
   - Adjust promotion gates and alert thresholds

5. **Test in paper trading**:
   - Run system with monitoring for 2-4 weeks
   - Validate alerts fire correctly
   - Compare backtest vs paper trading performance

---

## Troubleshooting

### CI Workflow Fails

**Issue**: Linting errors
- Run `ruff check src/ backtest/ --fix` locally
- Commit fixes

**Issue**: Type checking errors
- Run `mypy src/ backtest/ --ignore-missing-imports` locally
- Add type hints or `# type: ignore` comments

### Nightly Backtest Fails

**Issue**: Data fetch errors
- Check Alpaca API keys are set correctly
- Verify API is not rate-limited

**Issue**: Promotion gates always fail
- Review `logs/gate_summary.json` in artifacts
- Adjust thresholds in `config/promotion_gates.yaml`

### Monitoring Not Working

**Issue**: No metrics in Prometheus
- Check trading system is running and exporting metrics on port 9101
- Verify Prometheus can reach `localhost:9101`
- Check Prometheus targets: http://localhost:9090/targets

**Issue**: Dashboards show "No data"
- Verify Prometheus data source is configured correctly in Grafana
- Check metric names match between exporter and dashboard queries

---

## Support and Documentation

For more details, see:
- [docs/architecture.md](docs/architecture.md) - System architecture
- [docs/model-governance.md](docs/model-governance.md) - Model promotion workflow
- [docs/ops-playbook.md](docs/ops-playbook.md) - Daily operations procedures
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
