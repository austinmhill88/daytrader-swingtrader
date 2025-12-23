"""
Enhanced alerting for model drift, data quality, and performance anomalies (Phase 4).
Extends base alerting with ML-specific and data quality monitoring.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
from collections import deque

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - some drift detection features disabled")


class EnhancedAlertManager:
    """
    Enhanced alert manager for ML models, data quality, and performance monitoring.
    Phase 4 implementation for robust alerting.
    """
    
    def __init__(self, config: Dict, notifier=None):
        """
        Initialize enhanced alert manager.
        
        Args:
            config: Alert configuration
            notifier: Alert notifier instance
        """
        self.config = config
        self.notifier = notifier
        
        alerts_config = config.get('alerts', {})
        
        # Alert thresholds
        self.alert_on_model_drift = alerts_config.get('alert_on_model_drift', True)
        self.alert_on_data_stale = True
        self.alert_on_execution_lag = True
        
        # Model drift thresholds
        self.ks_threshold = 0.2  # Kolmogorov-Smirnov statistic
        self.psi_threshold = 0.3  # Population Stability Index
        
        # Data quality thresholds
        self.max_data_lag_minutes = 5
        self.min_data_points_per_symbol = 100
        self.max_missing_data_pct = 10.0
        
        # Performance anomaly detection
        self.performance_lookback_days = 20
        self.anomaly_std_threshold = 2.0  # Standard deviations
        
        # Alert history tracking
        self.alert_history = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_cooldown_minutes = 30  # Minimum time between duplicate alerts
        
        logger.info("EnhancedAlertManager initialized")
    
    def check_model_drift(
        self,
        model_name: str,
        feature_name: str,
        train_distribution: np.ndarray,
        prod_distribution: np.ndarray
    ) -> Optional[Dict]:
        """
        Check for model feature drift using statistical tests.
        
        Args:
            model_name: Model identifier
            feature_name: Feature name
            train_distribution: Training data distribution
            prod_distribution: Production data distribution
            
        Returns:
            Alert dict if drift detected, None otherwise
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, skipping drift detection")
            return None
        
        try:
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(train_distribution, prod_distribution)
            
            # Population Stability Index
            psi = self._calculate_psi(train_distribution, prod_distribution)
            
            drift_detected = ks_stat > self.ks_threshold or psi > self.psi_threshold
            
            if drift_detected and self._should_send_alert(f"drift_{model_name}_{feature_name}"):
                alert = {
                    'type': 'model_drift',
                    'severity': 'warning',
                    'model': model_name,
                    'feature': feature_name,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pval,
                    'psi': psi,
                    'timestamp': datetime.now().isoformat(),
                    'message': (
                        f"Model drift detected: {model_name} feature '{feature_name}' "
                        f"KS: {ks_stat:.3f}, PSI: {psi:.3f}"
                    )
                }
                
                self._send_alert(alert)
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking model drift: {e}")
            return None
    
    def check_data_staleness(
        self,
        symbol: str,
        last_update: datetime,
        expected_update_frequency: str = "1Min"
    ) -> Optional[Dict]:
        """
        Check if data is stale for a symbol.
        
        Args:
            symbol: Stock symbol
            last_update: Last data update timestamp
            expected_update_frequency: Expected update frequency
            
        Returns:
            Alert dict if data is stale, None otherwise
        """
        now = datetime.now()
        lag_minutes = (now - last_update).total_seconds() / 60.0
        
        if lag_minutes > self.max_data_lag_minutes:
            if self._should_send_alert(f"data_stale_{symbol}"):
                alert = {
                    'type': 'data_stale',
                    'severity': 'critical',
                    'symbol': symbol,
                    'lag_minutes': lag_minutes,
                    'last_update': last_update.isoformat(),
                    'timestamp': now.isoformat(),
                    'message': (
                        f"Data stale for {symbol}: Last update {lag_minutes:.1f} minutes ago "
                        f"(threshold: {self.max_data_lag_minutes} minutes)"
                    )
                }
                
                self._send_alert(alert)
                return alert
        
        return None
    
    def check_data_quality(
        self,
        symbol: str,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Optional[Dict]:
        """
        Check data quality issues.
        
        Args:
            symbol: Stock symbol
            df: DataFrame to check
            required_columns: Required columns
            
        Returns:
            Alert dict if quality issues found, None otherwise
        """
        issues = []
        
        # Check for missing required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for insufficient data
        if len(df) < self.min_data_points_per_symbol:
            issues.append(
                f"Insufficient data: {len(df)} rows "
                f"(minimum: {self.min_data_points_per_symbol})"
            )
        
        # Check for missing values
        for col in required_columns:
            if col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                if missing_pct > self.max_missing_data_pct:
                    issues.append(
                        f"High missing values in '{col}': {missing_pct:.1f}%"
                    )
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Duplicate timestamps: {duplicates}")
        
        # Check for anomalous values (e.g., negative prices)
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                issues.append("Invalid prices detected (<=0)")
        
        if issues and self._should_send_alert(f"data_quality_{symbol}"):
            alert = {
                'type': 'data_quality',
                'severity': 'critical',
                'symbol': symbol,
                'issues': issues,
                'rows': len(df),
                'timestamp': datetime.now().isoformat(),
                'message': (
                    f"Data quality issues for {symbol}: {'; '.join(issues)}"
                )
            }
            
            self._send_alert(alert)
            return alert
        
        return None
    
    def check_execution_lag(
        self,
        order_placement_time: datetime,
        fill_time: datetime,
        max_lag_seconds: float = 5.0
    ) -> Optional[Dict]:
        """
        Check for excessive execution lag.
        
        Args:
            order_placement_time: When order was placed
            fill_time: When order was filled
            max_lag_seconds: Maximum acceptable lag
            
        Returns:
            Alert dict if excessive lag detected, None otherwise
        """
        lag_seconds = (fill_time - order_placement_time).total_seconds()
        
        if lag_seconds > max_lag_seconds:
            if self._should_send_alert("execution_lag"):
                alert = {
                    'type': 'execution_lag',
                    'severity': 'warning',
                    'lag_seconds': lag_seconds,
                    'threshold_seconds': max_lag_seconds,
                    'timestamp': datetime.now().isoformat(),
                    'message': (
                        f"High execution lag: {lag_seconds:.2f}s "
                        f"(threshold: {max_lag_seconds}s)"
                    )
                }
                
                self._send_alert(alert)
                return alert
        
        return None
    
    def check_performance_anomaly(
        self,
        strategy_name: str,
        daily_returns: pd.Series
    ) -> Optional[Dict]:
        """
        Check for performance anomalies using statistical methods.
        
        Args:
            strategy_name: Strategy identifier
            daily_returns: Series of daily returns
            
        Returns:
            Alert dict if anomaly detected, None otherwise
        """
        if len(daily_returns) < self.performance_lookback_days:
            return None
        
        # Use recent lookback window
        recent_returns = daily_returns.tail(self.performance_lookback_days)
        
        # Calculate z-score of latest return
        mean_return = recent_returns[:-1].mean()
        std_return = recent_returns[:-1].std()
        latest_return = recent_returns.iloc[-1]
        
        if std_return > 0:
            z_score = (latest_return - mean_return) / std_return
            
            # Alert if return is anomalous (beyond threshold)
            if abs(z_score) > self.anomaly_std_threshold:
                if self._should_send_alert(f"performance_anomaly_{strategy_name}"):
                    alert = {
                        'type': 'performance_anomaly',
                        'severity': 'warning',
                        'strategy': strategy_name,
                        'z_score': z_score,
                        'latest_return': latest_return,
                        'mean_return': mean_return,
                        'std_return': std_return,
                        'timestamp': datetime.now().isoformat(),
                        'message': (
                            f"Performance anomaly for {strategy_name}: "
                            f"Return {latest_return:.2%} is {z_score:.1f} std devs "
                            f"from mean ({mean_return:.2%})"
                        )
                    }
                    
                    self._send_alert(alert)
                    return alert
        
        return None
    
    def check_data_gap(
        self,
        symbol: str,
        timestamps: pd.Series,
        expected_frequency: str = "1Min"
    ) -> Optional[Dict]:
        """
        Check for gaps in time series data.
        
        Args:
            symbol: Stock symbol
            timestamps: Series of timestamps
            expected_frequency: Expected data frequency
            
        Returns:
            Alert dict if gaps detected, None otherwise
        """
        if len(timestamps) < 2:
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)
        
        # Calculate time differences
        time_diffs = timestamps.diff()
        
        # Expected frequency in timedelta
        freq_map = {
            '1Min': timedelta(minutes=1),
            '5Min': timedelta(minutes=5),
            '15Min': timedelta(minutes=15),
            '1Hour': timedelta(hours=1),
            '1Day': timedelta(days=1)
        }
        expected_diff = freq_map.get(expected_frequency, timedelta(minutes=1))
        
        # Find gaps (>2x expected frequency)
        gaps = time_diffs[time_diffs > expected_diff * 2]
        
        if len(gaps) > 0 and self._should_send_alert(f"data_gap_{symbol}"):
            alert = {
                'type': 'data_gap',
                'severity': 'warning',
                'symbol': symbol,
                'num_gaps': len(gaps),
                'max_gap_minutes': gaps.max().total_seconds() / 60.0,
                'timestamp': datetime.now().isoformat(),
                'message': (
                    f"Data gaps detected for {symbol}: {len(gaps)} gaps, "
                    f"max gap: {gaps.max().total_seconds()/60.0:.1f} minutes"
                )
            }
            
            self._send_alert(alert)
            return alert
        
        return None
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Expected distribution (training)
            actual: Actual distribution (production)
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins from expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 2:
            return 0.0
        
        # Calculate proportions
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return psi
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """
        Check if alert should be sent (respects cooldown period).
        
        Args:
            alert_key: Unique key for alert type
            
        Returns:
            True if alert should be sent
        """
        now = datetime.now()
        
        if alert_key in self.last_alert_times:
            last_time = self.last_alert_times[alert_key]
            minutes_since = (now - last_time).total_seconds() / 60.0
            
            if minutes_since < self.alert_cooldown_minutes:
                logger.debug(
                    f"Alert '{alert_key}' suppressed (cooldown: "
                    f"{minutes_since:.1f}/{self.alert_cooldown_minutes} min)"
                )
                return False
        
        self.last_alert_times[alert_key] = now
        return True
    
    def _send_alert(self, alert: Dict) -> None:
        """
        Send alert via notifier.
        
        Args:
            alert: Alert dictionary
        """
        self.alert_history.append(alert)
        
        if self.notifier:
            try:
                severity = alert.get('severity', 'info')
                message = alert.get('message', 'Alert')
                
                if severity == 'critical':
                    self.notifier.send_critical_alert(message, alert)
                elif severity == 'warning':
                    self.notifier.send_warning_alert(message, alert)
                else:
                    self.notifier.send_info_alert(message, alert)
                    
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
        
        logger.info(f"Alert sent: {alert['type']} - {alert.get('message', '')}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of recent alerts.
        
        Args:
            hours: Hours to look back
            
        Returns:
            Summary dictionary
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]
        
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'hours': hours,
            'by_type': type_counts,
            'by_severity': severity_counts,
            'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
        }
