"""
Alert notifier for Slack, email, and other channels.
"""
import requests
import yaml
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict
import os


class AlertNotifier:
    """
    Handles sending alerts to various notification channels (Slack, email, etc.).
    Includes throttling and deduplication to prevent alert spam.
    """
    
    def __init__(self, alerts_config_path: str = "config/alerts.yaml"):
        """
        Initialize alert notifier.
        
        Args:
            alerts_config_path: Path to alerts configuration file
        """
        # Load alerts configuration
        try:
            with open(alerts_config_path, 'r') as f:
                self.alerts_config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load alerts config from {alerts_config_path}: {e}")
            self.alerts_config = {}
        
        # Get channel configurations
        channels = self.alerts_config.get('channels', {})
        self.slack_config = channels.get('slack', {})
        self.email_config = channels.get('email', {})
        
        # Resolve environment variables in webhook URL
        slack_webhook = self.slack_config.get('webhook_url', '')
        if slack_webhook.startswith('${') and slack_webhook.endswith('}'):
            env_var = slack_webhook[2:-1]
            self.slack_webhook = os.environ.get(env_var)
        else:
            self.slack_webhook = slack_webhook
        
        self.slack_enabled = self.slack_config.get('enabled', False) and self.slack_webhook
        self.email_enabled = self.email_config.get('enabled', False)
        
        # Throttling configuration
        throttling = self.alerts_config.get('throttling', {})
        self.throttling_enabled = throttling.get('enabled', True)
        self.max_alerts_per_minute = throttling.get('max_alerts_per_minute', 10)
        self.max_alerts_per_hour = throttling.get('max_alerts_per_hour', 50)
        self.dedupe_window_seconds = throttling.get('dedupe_window_seconds', 300)
        
        # Alert history for throttling and deduplication
        self.recent_alerts: List[datetime] = []
        self.alert_hashes: Dict[str, datetime] = {}
        
        logger.info(
            f"AlertNotifier initialized | "
            f"Slack: {self.slack_enabled}, Email: {self.email_enabled}"
        )
    
    def _check_throttle(self) -> bool:
        """
        Check if we're within throttling limits.
        
        Returns:
            True if we can send alert, False if throttled
        """
        if not self.throttling_enabled:
            return True
        
        now = datetime.now()
        
        # Clean old alerts
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self.recent_alerts = [t for t in self.recent_alerts if t > hour_ago]
        
        # Check limits
        alerts_last_minute = sum(1 for t in self.recent_alerts if t > minute_ago)
        alerts_last_hour = len(self.recent_alerts)
        
        if alerts_last_minute >= self.max_alerts_per_minute:
            logger.warning(f"Alert throttled: {alerts_last_minute} alerts in last minute")
            return False
        
        if alerts_last_hour >= self.max_alerts_per_hour:
            logger.warning(f"Alert throttled: {alerts_last_hour} alerts in last hour")
            return False
        
        return True
    
    def _check_dedupe(self, alert_hash: str) -> bool:
        """
        Check if this alert is a duplicate.
        
        Args:
            alert_hash: Hash of the alert content
            
        Returns:
            True if alert is new, False if duplicate
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.dedupe_window_seconds)
        
        # Clean old hashes
        self.alert_hashes = {
            h: t for h, t in self.alert_hashes.items()
            if t > cutoff
        }
        
        if alert_hash in self.alert_hashes:
            logger.debug(f"Duplicate alert suppressed: {alert_hash}")
            return False
        
        self.alert_hashes[alert_hash] = now
        return True
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Send an alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity ("info", "warning", "critical")
            metadata: Additional metadata
            
        Returns:
            True if alert sent successfully
        """
        # Create alert hash for deduplication
        alert_hash = f"{title}:{message}"
        
        # Check deduplication
        if not self._check_dedupe(alert_hash):
            return False
        
        # Check throttling
        if not self._check_throttle():
            return False
        
        # Record alert
        self.recent_alerts.append(datetime.now())
        
        success = True
        
        # Send to Slack
        if self.slack_enabled:
            try:
                self._send_slack(title, message, severity, metadata)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")
                success = False
        
        # Send to email
        if self.email_enabled:
            try:
                self._send_email(title, message, severity, metadata)
            except Exception as e:
                logger.error(f"Failed to send email alert: {e}")
                success = False
        
        return success
    
    def _send_slack(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Optional[Dict]
    ) -> None:
        """
        Send alert to Slack.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
        """
        if not self.slack_webhook:
            return
        
        # Map severity to color
        severity_config = self.alerts_config.get('severity_levels', {})
        color_map = {
            'info': severity_config.get('info', {}).get('color', '#36a64f'),
            'warning': severity_config.get('warning', {}).get('color', '#ff9900'),
            'critical': severity_config.get('critical', {}).get('color', '#ff0000')
        }
        color = color_map.get(severity, '#808080')
        
        # Build Slack message
        fields = []
        if metadata:
            for key, value in metadata.items():
                fields.append({
                    'title': key,
                    'value': str(value),
                    'short': True
                })
        
        payload = {
            'attachments': [{
                'color': color,
                'title': title,
                'text': message,
                'fields': fields,
                'footer': 'Trading System',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        # Add mention for critical alerts
        if severity == 'critical':
            mention = self.slack_config.get('mention_on_critical', '')
            if mention:
                payload['text'] = mention
        
        response = requests.post(
            self.slack_webhook,
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        
        logger.debug(f"Slack alert sent: {title}")
    
    def _send_email(
        self,
        title: str,
        message: str,
        severity: str,
        metadata: Optional[Dict]
    ) -> None:
        """
        Send alert via email.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
            
        Note:
            Email implementation is not yet complete. This is a placeholder
            for future SMTP integration. Enable email alerts only when
            implementation is complete.
        """
        # TODO: Implement SMTP email sending
        # Consider using smtplib or sendgrid/mailgun for production
        logger.info(f"Email alert (not implemented): {title} - {message}")
    
    def send_kill_switch_alert(self, reason: str, drawdown_pct: Optional[float] = None) -> None:
        """
        Send kill-switch activation alert.
        
        Args:
            reason: Reason for kill-switch activation
            drawdown_pct: Current drawdown percentage
        """
        metadata = {'reason': reason}
        if drawdown_pct is not None:
            metadata['drawdown_pct'] = f"{drawdown_pct:.2f}%"
        
        self.send_alert(
            title="🚨 KILL-SWITCH ACTIVATED",
            message=f"Trading halted: {reason}",
            severity="critical",
            metadata=metadata
        )
    
    def send_daily_pnl_alert(self, pnl_pct: float, pnl_usd: float) -> None:
        """
        Send daily P&L threshold alert.
        
        Args:
            pnl_pct: P&L percentage
            pnl_usd: P&L in USD
        """
        self.send_alert(
            title="Daily P&L Update",
            message=f"P&L: {pnl_pct:+.2f}% (${pnl_usd:,.2f})",
            severity="info",
            metadata={
                'pnl_pct': f"{pnl_pct:+.2f}%",
                'pnl_usd': f"${pnl_usd:,.2f}"
            }
        )
    
    def send_stream_disconnect_alert(self, disconnect_seconds: int) -> None:
        """
        Send data stream disconnect alert.
        
        Args:
            disconnect_seconds: Duration of disconnect in seconds
        """
        self.send_alert(
            title="⚠️ Data Stream Disconnected",
            message=f"Data feed disconnected for {disconnect_seconds}s",
            severity="warning",
            metadata={'disconnect_duration': f"{disconnect_seconds}s"}
        )
    
    def send_high_rejection_rate_alert(self, rejection_pct: float, lookback_minutes: int) -> None:
        """
        Send high order rejection rate alert.
        
        Args:
            rejection_pct: Rejection rate percentage
            lookback_minutes: Lookback period in minutes
        """
        self.send_alert(
            title="⚠️ High Order Rejection Rate",
            message=f"Rejection rate: {rejection_pct:.1f}% in last {lookback_minutes} minutes",
            severity="warning",
            metadata={
                'rejection_rate': f"{rejection_pct:.1f}%",
                'lookback_minutes': lookback_minutes
            }
        )
    
    def send_promotion_gate_failure_alert(self, model_id: str, strategy: str, failed_gates: List[str]) -> None:
        """
        Send model promotion gate failure alert.
        
        Args:
            model_id: Model identifier
            strategy: Strategy name
            failed_gates: List of failed gate names
        """
        self.send_alert(
            title="⚠️ Model Promotion Failed",
            message=f"Model {model_id} for {strategy} failed gates: {', '.join(failed_gates)}",
            severity="warning",
            metadata={
                'model_id': model_id,
                'strategy': strategy,
                'failed_gates': ', '.join(failed_gates)
            }
        )
