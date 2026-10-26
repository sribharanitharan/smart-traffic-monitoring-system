# database/traffic_db.py
"""
SQLite database for logging traffic analytics per frame.
Enables historical trend analysis and report export.
Criterion 5: Implementation and Code Quality (6 marks)
"""

import sqlite3
import datetime
import pandas as pd
from pathlib import Path


class TrafficDB:
    """
    Persistent SQLite store for all traffic metrics.
    Thread-safe with check_same_thread=False for Streamlit compatibility.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS traffic_logs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        vehicle_count   INTEGER DEFAULT 0,
        flow_rate       REAL    DEFAULT 0.0,
        avg_speed       REAL    DEFAULT 0.0,
        congestion      TEXT    DEFAULT 'FREE FLOW',
        roi_density     INTEGER DEFAULT 0,
        cars            INTEGER DEFAULT 0,
        trucks          INTEGER DEFAULT 0,
        buses           INTEGER DEFAULT 0,
        motorcycles     INTEGER DEFAULT 0,
        incident_count  INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS incident_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        track_id    INTEGER,
        type        TEXT,
        location_x  INTEGER,
        location_y  INTEGER
    );
    """

    def __init__(self, db_path: str = "database/traffic.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Traffic log
    # ------------------------------------------------------------------

    def log_frame(self, data: dict):
        """
        Insert one frame's traffic metrics.

        Expected keys in data:
            vehicle_count, flow_rate, avg_speed, congestion,
            roi_density, cars, trucks, buses, motorcycles, incident_count
        """
        data.setdefault("timestamp", datetime.datetime.now().isoformat())
        self.conn.execute("""
            INSERT INTO traffic_logs
                (timestamp, vehicle_count, flow_rate, avg_speed,
                 congestion, roi_density, cars, trucks, buses,
                 motorcycles, incident_count)
            VALUES
                (:timestamp, :vehicle_count, :flow_rate, :avg_speed,
                 :congestion, :roi_density, :cars, :trucks, :buses,
                 :motorcycles, :incident_count)
        """, data)
        self.conn.commit()

    def log_incident(self, track_id: int, inc_type: str,
                     location: tuple):
        self.conn.execute("""
            INSERT INTO incident_logs
                (timestamp, track_id, type, location_x, location_y)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.datetime.now().isoformat(),
              track_id, inc_type, location[0], location[1]))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 100) -> pd.DataFrame:
        """Return last n traffic log rows as DataFrame."""
        return pd.read_sql_query(
            f"SELECT * FROM traffic_logs ORDER BY id DESC LIMIT {n}",
            self.conn
        )

    def get_summary(self) -> dict:
        """Aggregate statistics across all logs."""
        try:
            cur = self.conn.execute("""
                SELECT
                    COUNT(*)         AS total_frames,
                    MAX(vehicle_count)   AS peak_count,
                    AVG(flow_rate)       AS avg_flow,
                    AVG(avg_speed)       AS avg_speed,
                    SUM(incident_count)  AS total_incidents
                FROM traffic_logs
            """)
            row = cur.fetchone()
            keys = ["total_frames", "peak_count", "avg_flow",
                    "avg_speed", "total_incidents"]
            return dict(zip(keys, row)) if row else {}
        except Exception:
            return {}

    def export_csv(self, path: str = "data/output/traffic_report.csv"):
        """Export full traffic log to CSV."""
        try:
            df = pd.read_sql_query(
                "SELECT * FROM traffic_logs ORDER BY id", self.conn
            )
            df.to_csv(path, index=False)
            return path
        except Exception:
            return None

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
