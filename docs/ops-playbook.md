# Operations Playbook

## Overview
This document provides operational procedures for daily/weekly routines, incident response, failover procedures, disaster recovery, and root-cause analysis templates.

---

## Daily Routines

### Pre-Market (08:00 - 09:30 ET)

#### 08:00 - System Health Check
**Purpose**: Verify all components are operational before market open

**Tasks**:
1. **Check System Status**
   ```bash
   # SSH to trading machine
   ssh trading@trading-pc
   
   # Check if trading system is running
   ps aux | grep python | grep main.py
   docker ps  # If using Docker
   ```

2. **Verify Data Feed Connection**
   - Check Alpaca WebSocket connection status
   - Verify last data timestamp is recent (< 5 minutes old)
   ```python
   system.data_feed.get_connection_status()
   # Expected: {'status': 'connected', 'last_message': '2024-01-15 08:00:05'}
   ```

3. **Check Disk Space**
   ```bash
   df -h
   # Ensure > 10% free on data partition
   ```

4. **Check Logs for Errors**
   ```bash
   tail -100 logs/errors.log
   # Look for any errors or warnings from overnight
   ```

5. **Verify Broker Account Status**
   ```python
   account = api.get_account()
   print(f"Account Status: {account.status}")
   print(f"Buying Power: ${account.buying_power}")
   print(f"Pattern Day Trader: {account.pattern_day_trader}")
   # Ensure account is active, not restricted
   ```

---

#### 08:30 - Pre-Market Data Sync and Universe Refresh
**Purpose**: Update universe and validate model freshness

**Tasks**:
1. **Run Scheduled Pre-Market Tasks**
   ```python
   system.scheduler.run_task('pre_market')
   # This triggers:
   # - Data sync (fetch previous day's bars)
   # - Universe refresh (liquidity, shortability, earnings)
   # - Model freshness check
   ```

2. **Review Universe Changes**
   ```bash
   # Check universe log
   cat logs/universe_changes.log
   # Look for significant additions/removals
   ```

3. **Validate Active Models**
   ```python
   models = system.ml_trainer.get_active_models()
   for model in models:
       print(f"Model: {model.id}, Age: {model.age_days} days, Status: {model.status}")
   # Ensure models are < 90 days old and status = 'active'
   ```

4. **Check Corporate Actions**
   ```python
   actions = system.data_sources.get_corporate_actions(date='today')
   for action in actions:
       print(f"{action.symbol}: {action.type} on {action.date}")
   # Look for splits, dividends, mergers
   ```

---

#### 09:00 - Final Pre-Market Checks
**Purpose**: Last-minute validation before trading starts

**Tasks**:
1. **Review Risk Limits**
   ```python
   risk_config = system.risk_manager.get_current_limits()
   print(f"Daily Max Drawdown: {risk_config['daily_max_drawdown_pct']}%")
   print(f"Max Gross Exposure: {risk_config['max_gross_exposure_pct']}%")
   print(f"Kill Switch: {risk_config['kill_switch']}")
   ```

2. **Check for Overnight Gaps**
   ```python
   positions = system.portfolio.get_positions()
   for pos in positions:
       overnight_pnl = calculate_overnight_pnl(pos)
       if abs(overnight_pnl) > 0.05:  # > 5% gap
           print(f"WARNING: {pos.symbol} gapped {overnight_pnl:.1%} overnight")
   ```

3. **Verify Admin Controls State**
   ```python
   state = system.admin_controls.get_state()
   print(f"Trading Paused: {state['is_paused']}")
   print(f"Manual Intervention: {state['manual_mode']}")
   # Ensure trading is not paused unless intentional
   ```

4. **Set Market Regime (if manual)**
   ```python
   # If using manual regime detection
   regime = detect_current_regime()
   system.regime_detector.set_regime(regime)
   print(f"Current Regime: {regime}")
   ```

---

### Market Hours (09:30 - 16:00 ET)

#### Continuous Monitoring

**Monitor Every 15-30 Minutes**:
1. **System Health**
   ```python
   health = system.self_healing.get_system_health()
   for component, status in health.items():
       if status != 'healthy':
           print(f"WARNING: {component} is {status}")
   ```

2. **Daily P&L**
   ```python
   pnl = system.portfolio.calculate_pnl()
   print(f"Daily P&L: ${pnl['unrealized'] + pnl['realized']:.2f} ({pnl['pct']:.2f}%)")
   print(f"Drawdown from HWM: {pnl['drawdown_pct']:.2f}%")
   ```

3. **Open Positions**
   ```python
   positions = system.portfolio.get_positions()
   print(f"Open Positions: {len(positions)}")
   print(f"Gross Exposure: {system.portfolio.get_exposures()['gross_pct']:.1f}%")
   ```

4. **Recent Orders and Fills**
   ```python
   recent_orders = system.execution_engine.get_recent_orders(minutes=30)
   fill_rate = sum([1 for o in recent_orders if o.status == 'filled']) / len(recent_orders)
   print(f"Fill Rate (last 30 min): {fill_rate:.1%}")
   ```

5. **Data Feed Latency**
   ```python
   latency = system.metrics_tracker.get_latency_summary(minutes=15)
   print(f"Median Data Latency: {latency['median']}ms")
   print(f"P95 Data Latency: {latency['p95']}ms")
   # Alert if P95 > 500ms
   ```

---

### End of Day (16:00 - 17:00 ET)

#### 16:00 - Market Close
**Purpose**: Ensure intraday positions are flattened

**Tasks**:
1. **Verify Intraday Flatten**
   ```python
   intraday_positions = system.portfolio.get_positions(strategy_type='intraday')
   if len(intraday_positions) > 0:
       print(f"WARNING: {len(intraday_positions)} intraday positions not flattened")
       # Manually flatten if needed
       system.admin_controls.flatten_positions(strategy='intraday')
   ```

---

#### 16:30 - End-of-Day Tasks
**Purpose**: Generate reports, reconcile, and backup data

**Tasks**:
1. **Run Scheduled EOD Tasks**
   ```python
   system.scheduler.run_task('end_of_day')
   # This triggers:
   # - Generate daily reports
   # - Backup data
   # - Reconcile positions with broker
   ```

2. **Review Daily Performance**
   ```bash
   cat logs/daily_report_2024-01-15.txt
   # Review P&L, trades, fills, slippage
   ```

3. **Reconcile Positions**
   ```python
   system.portfolio.reconcile_with_broker()
   # Compare internal positions vs broker positions
   # Alert if discrepancies found
   ```

4. **Archive Logs**
   ```bash
   # Compress and archive today's logs
   tar -czf logs/archive/logs_2024-01-15.tar.gz logs/*.log
   ```

5. **Check for Alerts**
   ```bash
   # Review Slack/email for any alerts during the day
   # Investigate any issues
   ```

---

## Weekly Routines

### Sunday Evening or Monday Pre-Market

**Purpose**: Weekly universe refresh and model review

**Tasks**:
1. **Full Universe Rebuild**
   ```python
   system.universe_analytics.rebuild_universe(force=True)
   # Recompute ADV, spreads, tiers
   ```

2. **Review Model Performance**
   ```python
   models = system.ml_trainer.get_active_models()
   for model in models:
       live_perf = get_live_performance(model, days=7)
       print(f"Model: {model.id}, 7-Day Sharpe: {live_perf['sharpe']:.2f}")
   # Flag models with degrading performance
   ```

3. **Review Weekly P&L**
   ```python
   weekly_pnl = get_weekly_pnl()
   print(f"Weekly P&L: ${weekly_pnl['total']:.2f} ({weekly_pnl['pct']:.2f}%)")
   print(f"Max Drawdown: {weekly_pnl['max_dd']:.2f}%")
   ```

4. **Check System Resource Usage**
   ```bash
   # CPU, Memory, Disk over past week
   # Look for any trends (memory leaks, disk filling up)
   ```

5. **Update Risk Limits (if needed)**
   ```python
   # Adjust risk limits based on performance and market conditions
   # Example: Increase daily DD limit from 2% to 3% if consistent profitability
   ```

---

## Incident Response

### Level 1: Minor Issue (Self-Healing Can Recover)

**Examples**:
- Data feed disconnection (< 2 minutes)
- Single order rejection
- Transient API error

**Action**: Monitor self-healing recovery
```python
# Self-healing will automatically attempt recovery
# Monitor logs and alerts
tail -f logs/self_healing.log
```

---

### Level 2: Moderate Issue (Manual Intervention Recommended)

**Examples**:
- Data feed disconnection (> 2 minutes)
- Multiple order rejections
- Performance degradation (latency > 1s)

**Action**:
1. **Pause Trading** (stop new entries, allow exits)
   ```python
   system.admin_controls.pause_trading(reason="Data feed issue")
   ```

2. **Investigate Root Cause**
   ```bash
   # Check logs
   tail -200 logs/errors.log
   
   # Check network connectivity
   ping alpaca.markets
   
   # Check broker API status
   curl https://status.alpaca.markets/api/v2/status.json
   ```

3. **Attempt Recovery**
   - Reconnect data feed
   - Retry failed orders
   - Clear any stuck states

4. **Resume Trading** (if issue resolved)
   ```python
   system.admin_controls.resume_trading(confirm=True)
   ```

---

### Level 3: Critical Issue (Emergency Halt)

**Examples**:
- Kill-switch triggered (daily drawdown breached)
- Broker API completely down
- Critical system failure (out of memory, disk full)
- Rogue orders detected

**Action**:
1. **Trigger Emergency Halt**
   ```python
   system.admin_controls.emergency_halt(reason="Kill-switch triggered: -2.1% daily DD")
   # This will:
   # - Stop all new orders
   # - Attempt to flatten all positions (if possible)
   # - Disconnect data feeds
   # - Log incident
   ```

2. **Manually Flatten Positions** (if system cannot)
   ```python
   # Use broker's web interface or API directly
   # Flatten all positions manually
   ```

3. **Notify Team**
   - Send alert to Slack/email
   - Include incident details and actions taken

4. **Root Cause Analysis** (see section below)

5. **Do Not Resume** until root cause is resolved and validated

---

## Failover Procedures

### Data Feed Failover (Alpaca → Backup Source)

**Scenario**: Alpaca data feed is down or delayed

**Action**:
1. **Automatic Failover** (if configured)
   ```python
   # System should automatically switch to backup source
   # Monitor logs
   tail -f logs/data_feed.log
   # Look for: "Failover to backup source: polygon"
   ```

2. **Manual Failover** (if needed)
   ```python
   system.data_sources.switch_to_backup(source='polygon')
   # Or
   system.data_sources.switch_to_backup(source='tiingo')
   ```

3. **Validate Data Quality**
   ```python
   # Check that data is flowing
   latest_bars = system.data_feed.get_latest_bars(['AAPL', 'MSFT'])
   for symbol, bar in latest_bars.items():
       print(f"{symbol}: {bar.timestamp}, Close: ${bar.close}")
   ```

4. **Resume Trading** (if data quality is good)

---

### Broker API Failover (REST → WebSocket or Vice Versa)

**Scenario**: REST API is slow or unresponsive

**Action**:
1. **Switch to WebSocket** (if available)
   ```python
   # Use WebSocket for order submission and account updates
   system.alpaca_client.switch_to_websocket()
   ```

2. **Reduce Request Rate**
   - Throttle order submissions
   - Reduce polling frequency

3. **Contact Broker Support** (if issue persists)

---

## Disaster Recovery

### Scenario 1: Local PC Failure (Hardware Issue)

**Impact**: System is completely down

**Recovery Steps**:
1. **Flatten Positions** (if possible)
   - Use backup machine or broker's web interface
   - Flatten all open positions

2. **Restore System on Backup Machine**
   ```bash
   # Restore from latest backup
   rsync -av backup:/data/backups/latest/ /home/trading/
   
   # Start system
   docker-compose up -d
   ```

3. **Reconcile State**
   ```python
   system.portfolio.reconcile_with_broker()
   ```

4. **Resume Trading** (after validation)

**Prevention**:
- Keep backup machine ready (or use cloud)
- Daily backups to external storage
- Test disaster recovery quarterly

---

### Scenario 2: Data Loss (Disk Failure)

**Impact**: Historical data and model artifacts lost

**Recovery Steps**:
1. **Restore from Backup**
   ```bash
   # Restore from latest backup (external SSD or cloud)
   rsync -av /mnt/backup/data/ /home/trading/data/
   ```

2. **Re-fetch Missing Data** (if backup is old)
   ```python
   # Fetch missing bars from broker API
   system.data_storage.backfill_missing_data(
       start_date='2024-01-01',
       end_date='2024-01-15'
   )
   ```

3. **Validate Data Integrity**
   ```python
   system.data_storage.validate_data(
       symbols=['AAPL', 'MSFT', ...],
       date_range=('2024-01-01', '2024-01-15')
   )
   ```

4. **Re-train Models** (if model artifacts lost)
   ```python
   system.ml_trainer.retrain_all_models()
   ```

**Prevention**:
- Daily backups to external storage
- Weekly backups to cloud (encrypted)
- RAID or redundant storage

---

### Scenario 3: Broker Outage (Alpaca Down)

**Impact**: Cannot trade or access account

**Recovery Steps**:
1. **Confirm Outage**
   - Check https://status.alpaca.markets/
   - Check Twitter/Reddit for reports

2. **Do Not Panic**
   - Open positions will remain open
   - Stop-loss orders (if placed at broker) should still trigger

3. **Wait for Restoration**
   - Monitor broker status page
   - Prepare to reconcile once restored

4. **Reconcile After Restoration**
   ```python
   system.portfolio.reconcile_with_broker()
   # Check for any fills that occurred during outage
   ```

5. **Review Incident**
   - Document impact (missed trades, slippage, etc.)
   - Consider multi-broker setup for future

**Prevention**:
- Have account with backup broker (Interactive Brokers, TD Ameritrade)
- Test backup broker connectivity quarterly

---

## Root-Cause Analysis Template

### Incident Report

**Date/Time**: 2024-01-15 14:35 ET

**Severity**: Level 3 (Critical)

**Summary**: Kill-switch triggered due to rapid drawdown

---

### Timeline

| Time | Event |
|------|-------|
| 14:30 | Market volatility spike (VIX +15%) |
| 14:32 | Multiple positions hit stop-losses |
| 14:35 | Daily P&L reached -2.1% (trigger: -2.0%) |
| 14:35 | Kill-switch activated, trading halted |
| 14:36 | Operator notified via Slack alert |
| 14:40 | Emergency halt confirmed, positions flattened |
| 14:45 | Investigation started |

---

### Root Cause

**Primary Cause**: Unexpected market volatility spike due to FOMC announcement

**Contributing Factors**:
- All strategies were net long during announcement
- Stop-losses were triggered simultaneously
- No regime detector to reduce exposure before announcement

---

### Impact

- **Financial**: -2.1% daily loss ($4,200 on $200K account)
- **Operational**: System halted for 2 hours during investigation
- **Reputational**: None (internal system)

---

### Actions Taken

1. Flattened all positions (14:40 ET)
2. Reviewed system logs and order history
3. Identified lack of event-based risk reduction
4. Documented lessons learned

---

### Preventive Measures

**Immediate** (within 1 week):
- [ ] Implement economic calendar integration
- [ ] Add pre-announcement exposure reduction (FOMC, CPI, etc.)
- [ ] Test new logic in paper trading

**Short-term** (within 1 month):
- [ ] Improve regime detector to detect pre-announcement volatility
- [ ] Add event-driven risk limits (reduce exposure 30 min before major events)

**Long-term** (within 3 months):
- [ ] Implement event-driven strategies that can profit from announcements
- [ ] Backtest performance during historical FOMC events

---

### Sign-off

**Reviewed by**: [Operator Name]
**Date**: 2024-01-15
**Status**: Closed

---

## Kill-Switch Procedures

### When Kill-Switch Triggers

**Automatic Triggers**:
1. Daily drawdown exceeds threshold (default: -2.0%)
2. Broker API errors (5 consecutive failures)
3. Data feed disconnection (> 2 minutes)

**Manual Trigger**:
```python
system.admin_controls.emergency_halt(reason="Manual intervention: market crash")
```

---

### Post-Kill-Switch Actions

1. **Assess Situation**
   - What triggered the kill-switch?
   - Are positions flattened?
   - Is system stable?

2. **Flatten Positions** (if not already)
   ```python
   system.admin_controls.flatten_all_positions()
   ```

3. **Review Logs**
   ```bash
   grep "ERROR" logs/runtime.log
   tail -500 logs/trades.log
   ```

4. **Root-Cause Analysis** (use template above)

5. **Do Not Resume** until:
   - Root cause is understood
   - Fix is implemented and tested
   - Risk limits are reviewed and adjusted if needed
   - Operator approval

---

## Monitoring Dashboards

### Grafana Dashboards (Recommended)

1. **Trading Overview**
   - Real-time P&L
   - Open positions
   - Gross/net exposure
   - Kill-switch status

2. **Execution Metrics**
   - Fill rate
   - Order latency
   - Slippage
   - Rejection rate

3. **System Health**
   - Data feed latency
   - CPU/Memory usage
   - Disk space
   - Component status

4. **Strategy Performance**
   - Per-strategy P&L
   - Win rate
   - Sharpe ratio (rolling)
   - Drawdown

---

## Contact Information

### Escalation Path

**Level 1: Self-Service**
- Check this playbook
- Review system logs
- Attempt standard recovery procedures

**Level 2: Operator**
- Contact: [Your Name]
- Phone: [Your Phone]
- Email: [Your Email]
- Slack: @operator

**Level 3: Emergency**
- Broker Support: 1-800-XXX-XXXX
- Cloud Provider Support: [If using cloud]
- Hardware Vendor Support: [If hardware issue]

---

## Appendix: Useful Commands

### System Status
```bash
# Check if system is running
ps aux | grep python | grep main.py
docker ps

# Check logs
tail -f logs/runtime.log
tail -f logs/errors.log

# Check disk space
df -h

# Check network
ping alpaca.markets
curl https://status.alpaca.markets/api/v2/status.json
```

### Data Operations
```python
# Re-fetch missing data
system.data_storage.backfill_missing_data('2024-01-01', '2024-01-15')

# Validate data
system.data_storage.validate_data(['AAPL', 'MSFT'], ('2024-01-01', '2024-01-15'))

# Create snapshot
system.data_storage.create_snapshot(name='pre_retrain_2024-01-15')
```

### Trading Controls
```python
# Pause trading
system.admin_controls.pause_trading(reason="Manual intervention")

# Resume trading
system.admin_controls.resume_trading(confirm=True)

# Emergency halt
system.admin_controls.emergency_halt(reason="Kill-switch triggered")

# Flatten positions
system.admin_controls.flatten_all_positions()
system.admin_controls.flatten_positions(strategy='intraday')
```

### Model Operations
```python
# Get active models
models = system.ml_trainer.get_active_models()

# Retrain model
system.ml_trainer.retrain_model(model_id='xyz123')

# Promote model
system.ml_trainer.promote_model(model_id='xyz123')

# Demote model
system.ml_trainer.demote_model(model_id='xyz123', reason="Poor live performance")
```

---

## Data Snapshots and Retention Policy

### Overview
Regular snapshots ensure reproducibility of backtests and provide disaster recovery capability. This section documents snapshot procedures and retention policies.

### Monthly Snapshot Procedures

**Schedule**: First Sunday of each month at 02:00 ET

**Tasks**:
1. **Create compressed snapshot**
   ```bash
   #!/bin/bash
   # Monthly snapshot script
   SNAPSHOT_DATE=$(date +%Y%m)
   SNAPSHOT_NAME="trading-snapshot-${SNAPSHOT_DATE}"
   
   # Create snapshot directory
   mkdir -p /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}
   
   # Copy raw data (parquet files)
   tar -I 'zstd -T0' -cf /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/raw-data.tar.zst \
     -C ${DATA_DIR:-./data} parquet/
   
   # Copy feature store
   tar -I 'zstd -T0' -cf /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/features.tar.zst \
     -C ${FEATURE_DIR:-./data/features} .
   
   # Copy models
   tar -I 'zstd -T0' -cf /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/models.tar.zst \
     -C ${MODEL_DIR:-./data/models} .
   
   # Copy reports
   tar -I 'zstd -T0' -cf /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/reports.tar.zst \
     -C ${LOGS_DIR:-./logs} .
   
   # Create manifest
   cat > /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/manifest.txt << EOF
   Snapshot: ${SNAPSHOT_NAME}
   Date: $(date -I)
   Raw Data: $(du -sh /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/raw-data.tar.zst | cut -f1)
   Features: $(du -sh /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/features.tar.zst | cut -f1)
   Models: $(du -sh /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/models.tar.zst | cut -f1)
   Reports: $(du -sh /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/reports.tar.zst | cut -f1)
   EOF
   
   echo "Snapshot ${SNAPSHOT_NAME} created successfully"
   ```

2. **Rsync to archive location**
   ```bash
   # Sync to 1TB archive SSD (when available)
   rsync -avz --progress \
     /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/ \
     /mnt/archive/trading-snapshots/${SNAPSHOT_NAME}/
   
   # Verify checksum
   cd /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}
   sha256sum *.tar.zst > checksums.sha256
   rsync -avz checksums.sha256 /mnt/archive/trading-snapshots/${SNAPSHOT_NAME}/
   ```

3. **Verify snapshot integrity**
   ```bash
   # Test extracting a sample
   tar -I zstd -tf /mnt/ssd2tb/snapshots/${SNAPSHOT_NAME}/raw-data.tar.zst | head -10
   
   # Verify checksums match
   cd /mnt/archive/trading-snapshots/${SNAPSHOT_NAME}
   sha256sum -c checksums.sha256
   ```

### Retention Policy

**Hot Data** (on 2TB SSD):
- Raw parquet data: Last 90 days
- Features: Last 60 days
- Models: Last 3 versions per strategy
- Logs: Last 30 days

**Cold Data** (on 1TB archive):
- Monthly snapshots: Keep all (or last 24 months)
- Compressed with zstd for space efficiency
- Verify checksums quarterly

**Cleanup Schedule**:
```bash
# Weekly cleanup (every Monday at 03:00 ET)
# Remove old hot data beyond retention windows

# Raw data > 90 days
find ${DATA_DIR:-./data}/parquet -type f -mtime +90 -delete

# Features > 60 days
find ${FEATURE_DIR:-./data/features} -type f -mtime +60 -delete

# Logs > 30 days (except error logs, keep 90 days)
find ${LOGS_DIR:-./logs} -name "*.log" -not -name "errors.log" -mtime +30 -delete
find ${LOGS_DIR:-./logs} -name "errors.log" -mtime +90 -delete

# Old model versions (keep only last 3)
python << 'PYTHON_CLEANUP'
import os
from pathlib import Path

model_dir = Path(os.environ.get('MODEL_DIR', './data/models'))
for strategy_dir in model_dir.iterdir():
    if strategy_dir.is_dir():
        models = sorted(strategy_dir.glob('*.pkl'), key=os.path.getmtime, reverse=True)
        # Keep latest 3, delete rest
        for old_model in models[3:]:
            old_model.unlink()
            print(f"Deleted old model: {old_model}")
PYTHON_CLEANUP
```

### Storage Budget Management

**Projected Usage** (for 200-symbol universe):
- Raw data: ~50GB per 90 days
- Features: ~20GB per 60 days
- Models: ~500MB per strategy
- Logs: ~5GB per 30 days
- **Total hot data**: ~80-100GB

**2TB SSD Allocation**:
- Hot data: 100GB
- Snapshots (3 months): 300GB
- Buffer: 1600GB free for future growth

**1TB Archive SSD** (future):
- Monthly snapshots: ~100GB each
- Can store ~24 months of history

### Restore Procedure

**To restore from snapshot**:
```bash
# Example: Restore data from December 2024 snapshot
SNAPSHOT_NAME="trading-snapshot-202412"
RESTORE_DATE=$(date +%Y%m%d_%H%M%S)

# Create restore directory
mkdir -p /tmp/restore_${RESTORE_DATE}

# Extract data
cd /mnt/archive/trading-snapshots/${SNAPSHOT_NAME}
tar -I zstd -xf raw-data.tar.zst -C /tmp/restore_${RESTORE_DATE}/
tar -I zstd -xf features.tar.zst -C /tmp/restore_${RESTORE_DATE}/
tar -I zstd -xf models.tar.zst -C /tmp/restore_${RESTORE_DATE}/

# Verify checksums
sha256sum -c checksums.sha256

# Copy to production locations
rsync -avz /tmp/restore_${RESTORE_DATE}/ ${DATA_DIR:-./data}/

echo "Restore from ${SNAPSHOT_NAME} complete"
```

### Automation with Cron

Add to crontab:
```bash
# Monthly snapshot (1st Sunday at 02:00 ET)
0 7 1-7 * 0 [ $(date +\%u) -eq 7 ] && /opt/trading/scripts/monthly_snapshot.sh >> /var/log/trading/snapshots.log 2>&1

# Weekly cleanup (every Monday at 03:00 ET)
0 8 * * 1 /opt/trading/scripts/weekly_cleanup.sh >> /var/log/trading/cleanup.log 2>&1

# Daily disk space check (every day at 06:00 ET)
0 11 * * * /opt/trading/scripts/check_disk_space.sh >> /var/log/trading/disk.log 2>&1
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | System | Initial version |
| 1.1 | 2024-12-22 | System | Added snapshot and retention policy section |

---

**Remember**: Stay calm during incidents. Follow procedures. Document everything.
