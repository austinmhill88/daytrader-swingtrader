# Data Governance and Reproducibility

## Overview
This document defines the data governance policies for the trading system, ensuring reproducible backtests, audit trails, and data quality standards.

## Data Versioning

### Versioning Strategy
- **Primary Method**: Dataset hashes (SHA-256) stored in model registry
- **Secondary Method**: Immutable snapshots with timestamp-based naming
- **Tool Options**: DVC (Data Version Control) or LakeFS for large datasets

### Version Metadata
Each data version includes:
- Dataset hash (SHA-256)
- Creation timestamp
- Data sources used (Alpaca, Polygon, Tiingo, IEX)
- Symbol universe (list of tickers)
- Date range covered
- Adjustment type (split-adjusted, dividend-adjusted, unadjusted)
- Known data gaps or anomalies

## Snapshot Management

### Snapshot Cadence
- **Daily snapshots**: Raw OHLCV data (hot storage, 90-day retention)
- **Weekly snapshots**: Feature store outputs (hot storage, 1-year retention)
- **Monthly snapshots**: Full historical archive (cold storage, permanent)
- **Pre-training snapshots**: Created before each model training run (permanent)

### Snapshot Naming Convention
```
snapshots/
├── raw/
│   ├── YYYY-MM-DD_raw_ohlcv_<hash>.parquet
│   └── YYYY-MM-DD_raw_ohlcv_<hash>.manifest.json
├── features/
│   ├── YYYY-MM-DD_features_<hash>.parquet
│   └── YYYY-MM-DD_features_<hash>.manifest.json
└── models/
    ├── YYYY-MM-DD_training_<model_id>_<hash>.parquet
    └── YYYY-MM-DD_training_<model_id>_<hash>.manifest.json
```

### Manifest Files
Each snapshot includes a JSON manifest:
```json
{
  "snapshot_id": "2024-01-15_features_a3f2c1b8",
  "created_at": "2024-01-15T02:00:00Z",
  "data_type": "features",
  "hash": "a3f2c1b8e4d9...",
  "sources": ["alpaca"],
  "symbols": ["AAPL", "MSFT", ...],
  "date_range": {"start": "2023-01-01", "end": "2024-01-14"},
  "row_count": 1234567,
  "columns": ["open", "high", "low", "close", "volume", ...],
  "quality_checks": {
    "missing_values": 0.001,
    "outliers_detected": 42,
    "gaps_filled": 3
  }
}
```

## Reproducibility Requirements

### Backtest Reproducibility
To reproduce a backtest result:
1. Record the dataset hash used
2. Store the exact model version (model registry ID)
3. Log random seeds for any stochastic components
4. Document the configuration file version (commit hash)
5. Capture Python/library versions (requirements.txt hash)

### Model Training Reproducibility
To reproduce a model training run:
1. Use the pre-training snapshot (immutable)
2. Record all hyperparameters in model registry
3. Log random seeds (Python, NumPy, model library)
4. Document train/validation split windows
5. Store feature engineering code commit hash

### Audit Trail
All reproducibility metadata is stored in:
- Model registry (MLflow tracking)
- Backtest results database
- Git commits (configuration and code)

## Storage Architecture

### Storage Tiers

#### Hot Storage (NVMe SSD - Samsung 990 Pro 2TB)
- **Purpose**: Active trading, recent data, feature computation
- **Contents**:
  - Last 90 days of raw OHLCV (minute bars)
  - Last 365 days of daily OHLCV
  - Current feature store outputs
  - Active model artifacts
  - Live trading logs
- **Retention**: 90 days for minute data, 1 year for daily data
- **Performance Target**: <10ms read latency for recent data

#### Cold Storage (External/NAS)
- **Purpose**: Historical archives, compliance, long-term backtests
- **Contents**:
  - Full historical OHLCV (years of data)
  - Archived feature snapshots (monthly)
  - Retired model artifacts
  - Archived backtest results
- **Retention**: Permanent (with compression)
- **Access Pattern**: Batch/offline only

### Partitioning Strategy
```
data/
├── parquet/
│   ├── ohlcv/
│   │   ├── symbol=AAPL/
│   │   │   ├── date=2024-01-01/
│   │   │   │   └── data.parquet
│   │   │   └── date=2024-01-02/
│   │   │       └── data.parquet
│   │   └── symbol=MSFT/
│   │       └── ...
│   └── features/
│       ├── symbol=AAPL/
│       │   ├── date=2024-01-01/
│       │   │   └── features.parquet
│       │   └── ...
│       └── ...
└── snapshots/
    ├── raw/
    ├── features/
    └── models/
```

### Compression Strategy
- **Format**: Parquet with Snappy compression
- **Typical compression ratio**: 5:1 to 10:1 for OHLCV data
- **Trade-off**: Balance between compression ratio and read performance
- **Alternative for cold storage**: Zstandard (better compression, slower)

## Data Quality Standards

### Validation Rules
1. **Price Sanity Checks**:
   - 0 < Low ≤ High
   - Low ≤ Open, Close ≤ High
   - Price changes < 50% per bar (except splits/halts)
   
2. **Volume Sanity Checks**:
   - Volume ≥ 0
   - Volume < 10x typical volume (flag outliers)
   
3. **Timestamp Checks**:
   - No duplicate timestamps per symbol
   - Timestamps in chronological order
   - Market hours alignment (except pre/post market)

4. **Completeness Checks**:
   - Flag missing bars during market hours
   - Track data source availability
   - Monitor data latency

### Data Correction Policy
- **Principle**: Never alter raw data; create versioned derived datasets
- **Process**:
  1. Flag anomalies in raw data
  2. Create corrected version with adjustments logged
  3. Document corrections in manifest
  4. Use corrected version for training/backtests
  5. Keep raw version for audit

### Corporate Actions Handling
- **Splits**: Apply split adjustment to historical prices
- **Dividends**: Optional dividend adjustment based on strategy type
- **Mergers/Acquisitions**: Flag affected symbols, handle delisting
- **Spin-offs**: Manual review and data reconciliation

## Data Source Priority

### Primary Source: Alpaca
- Real-time streaming during market hours
- Historical data retrieval for recent periods
- Free tier limitations: IEX data only

### Fallback Sources
1. **Polygon.io**: Premium data, higher quality, consolidated tape
2. **Tiingo**: Good for EOD data and corporate actions
3. **IEX Cloud**: Free tier available, suitable for testing

### Source Selection Logic
```python
if live_trading:
    use primary_source (Alpaca)
    on_error:
        fallback_to secondary_sources
else:  # backtesting
    prefer highest_quality_source (Polygon > Tiingo > Alpaca > IEX)
```

## Data Retention Policy

### Legal/Compliance Requirements
- Minimum 5 years retention for all trade-related data (U.S. regulations)
- Permanent retention recommended for model training datasets

### Practical Retention
- **Raw OHLCV**: Permanent (compressed in cold storage)
- **Features**: 2 years hot, permanent cold (monthly snapshots)
- **Model artifacts**: Permanent (until superseded and 1 year after)
- **Backtest results**: Permanent (lightweight, high audit value)
- **Logs**: 1 year for detailed logs, permanent for trade logs

### Cleanup/Archival Process
- Automated nightly job moves data from hot to cold storage
- Manual approval required for deletion of any data <5 years old
- Checksums verified during archival
- Restore procedures tested quarterly

## Data Access Controls

### Access Levels
1. **Read-only**: Backtesting, research, reporting
2. **Write (controlled)**: Data ingestion, feature computation
3. **Admin**: Manual corrections, schema changes, archival

### Audit Logging
- Log all data access with timestamp, user, operation
- Log all data modifications with before/after snapshots
- Retain audit logs permanently

## Disaster Recovery

### Backup Strategy
- **Frequency**: Daily incremental, weekly full
- **Locations**: 
  - Primary: Local NVMe
  - Secondary: External SSD
  - Tertiary: Cloud storage (optional, encrypted)
- **Encryption**: AES-256 for all backups
- **Testing**: Quarterly restore drills

### Recovery Time Objectives (RTO)
- **Critical data** (last 90 days): < 1 hour
- **Full historical data**: < 24 hours

### Recovery Point Objectives (RPO)
- **Streaming data**: Up to 1 hour of data loss acceptable (can be re-fetched)
- **Trade executions**: Zero tolerance (already logged to broker)
- **Model training**: Last snapshot (daily or pre-training)

## Implementation Checklist

- [ ] Implement SHA-256 hashing for all datasets
- [ ] Create snapshot generation scripts (daily/weekly/monthly)
- [ ] Set up manifest file generation
- [ ] Configure DVC or LakeFS (if using)
- [ ] Implement data quality validation pipeline
- [ ] Set up automated archival to cold storage
- [ ] Implement backup verification scripts
- [ ] Create data restoration procedures
- [ ] Set up audit logging for data access
- [ ] Document data source failover logic
- [ ] Test disaster recovery procedures

## References
- [SEC Rule 17a-4](https://www.sec.gov/rules/interp/34-47806.htm) - Record retention requirements
- [FINRA 4511](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4511) - General requirements
- [DVC Documentation](https://dvc.org/doc) - Data version control
- [LakeFS](https://lakefs.io/) - Data lake versioning
