# modules/alerts.py
"""
Alert management for congestion and incident notifications.
Supports console, log file, and optional email notifications.
Criterion 4: Computer Vision Method / AI Content (6 marks)
"""

import time
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


logger = logging.getLogger("traffic_alerts")


class AlertManager:
    """
    Rule-based alert system that:
    - Monitors congestion level changes
    - Receives incident notifications from IncidentDetector
    - Enforces cooldown to prevent spam
    - Logs all alerts with timestamps
    - Optionally sends email notifications
    """

    TRIGGER_LEVELS = {"CONGESTED", "HEAVY JAM"}

    def __init__(self,
                 congestion_cooldown_sec: int = 300,
                 incident_cooldown_sec:   int = 60,
                 email_config: dict = None):
        """
        Args:
            congestion_cooldown_sec : Min seconds between congestion alerts.
            incident_cooldown_sec   : Min seconds between incident alerts.
            email_config            : Optional dict with SMTP settings.
        """
        self.cong_cooldown  = congestion_cooldown_sec
        self.inc_cooldown   = incident_cooldown_sec
        self.email_config   = email_config

        self._last_cong_alert: float = 0
        self._last_inc_alert : dict  = {}   # incident_type → timestamp
        self.alert_log       : list  = []

    # ------------------------------------------------------------------

    def check_congestion(self,
                         level: str,
                         flow_rate: float,
                         location: str = "Main Road") -> list:
        """
        Check if congestion warrants an alert.

        Returns:
            List of alert message strings (empty if no alert).
        """
        alerts = []
        now = time.time()

        if (level in self.TRIGGER_LEVELS and
                now - self._last_cong_alert > self.cong_cooldown):

            msg = (f"⚠️  CONGESTION ALERT | Level: {level} | "
                   f"Flow: {flow_rate:.1f} veh/min | "
                   f"Location: {location} | "
                   f"Time: {time.strftime('%H:%M:%S')}")

            alerts.append(msg)
            self._last_cong_alert = now
            self._log(msg, "CONGESTION")

            if self.email_config:
                self._send_email(f"Traffic Alert: {level}", msg)

        return alerts

    def check_incidents(self, incidents: dict) -> list:
        """
        Check for new incident alerts.

        Args:
            incidents: {track_id: incident_type} from IncidentDetector.

        Returns:
            List of alert message strings.
        """
        alerts = []
        now = time.time()

        for tid, itype in incidents.items():
            last = self._last_inc_alert.get(itype, 0)
            if now - last > self.inc_cooldown:
                msg = (f"🚨 INCIDENT ALERT | Type: {itype} | "
                       f"Vehicle ID: {tid} | "
                       f"Time: {time.strftime('%H:%M:%S')}")
                alerts.append(msg)
                self._last_inc_alert[itype] = now
                self._log(msg, itype)

        return alerts

    # ------------------------------------------------------------------

    def _log(self, message: str, alert_type: str):
        """Save alert to in-memory log and Python logger."""
        self.alert_log.append({
            "time"    : time.strftime("%Y-%m-%d %H:%M:%S"),
            "type"    : alert_type,
            "message" : message,
        })
        logger.warning(message)
        print(f"[ALERT] {message}")

    def _send_email(self, subject: str, body: str):
        """Send email notification via SMTP (optional)."""
        try:
            cfg = self.email_config
            msg = MIMEMultipart()
            msg["From"]    = cfg["sender"]
            msg["To"]      = cfg["recipient"]
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL(cfg["smtp_host"], cfg.get("smtp_port", 465)) as server:
                server.login(cfg["sender"], cfg["password"])
                server.sendmail(cfg["sender"], cfg["recipient"],
                                msg.as_string())
        except Exception as e:
            logger.error(f"Email send failed: {e}")

    def get_recent_alerts(self, n: int = 10) -> list:
        """Return the n most recent alerts."""
        return self.alert_log[-n:]

    def get_log(self) -> list:
        return list(self.alert_log)
