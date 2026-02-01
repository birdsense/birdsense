"""
BirdSense Database Module
SQLite database for stats and configuration
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

DATABASE_FILE = '/data/birdsense.db'


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema"""
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()

        # Detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_nl TEXT NOT NULL,
                species_en TEXT,
                confidence INTEGER NOT NULL,
                camera TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                event_id TEXT,
                inference_time_ms INTEGER,
                image_path TEXT,
                thumbnail_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Add inference_time_ms column if it doesn't exist (migration)
        cursor.execute('PRAGMA table_info(detections)')
        columns = [row[1] for row in cursor.fetchall()]

        if 'inference_time_ms' not in columns:
            try:
                cursor.execute('ALTER TABLE detections ADD COLUMN inference_time_ms INTEGER')
                logger.info("Added inference_time_ms column")
            except sqlite3.OperationalError:
                pass

        # Add image_path column if it doesn't exist (migration)
        if 'image_path' not in columns:
            try:
                cursor.execute('ALTER TABLE detections ADD COLUMN image_path TEXT')
                logger.info("Added image_path column")
            except sqlite3.OperationalError:
                pass

        # Add thumbnail_path column if it doesn't exist (migration)
        if 'thumbnail_path' not in columns:
            try:
                cursor.execute('ALTER TABLE detections ADD COLUMN thumbnail_path TEXT')
                logger.info("Added thumbnail_path column")
            except sqlite3.OperationalError:
                pass

        # Create index for common queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
            ON detections(timestamp DESC)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_species
            ON detections(species_nl)
        ''')

        # Config table (key-value store)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Classification log table (for all classifications including unknowns)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_en TEXT,
                species_nl TEXT,
                confidence INTEGER NOT NULL,
                camera TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                top_predictions TEXT,
                image_path TEXT,
                status TEXT DEFAULT 'pending',
                species_corrected TEXT,
                corrected_at DATETIME,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create index for classification log queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_classification_log_timestamp
            ON classification_log(timestamp DESC)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_classification_log_status
            ON classification_log(status)
        ''')

        logger.info(f"Database initialized at {DATABASE_FILE}")


class BirdStats:
    """Manage bird detection statistics using SQLite"""

    @staticmethod
    def add_detection(species_nl: str, camera: str, confidence: int,
                      species_en: str = None, event_id: str = None,
                      timestamp: int = None, inference_time_ms: int = None,
                      image_path: str = None, thumbnail_path: str = None):
        """Add a new detection to the database"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (species_nl, species_en, confidence, camera, timestamp, event_id, inference_time_ms, image_path, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (species_nl, species_en, confidence, camera, timestamp, event_id, inference_time_ms, image_path, thumbnail_path))

        logger.debug(f"Detection saved: {species_nl} ({confidence}%) from {camera} in {inference_time_ms}ms")

    @staticmethod
    def get_total_detections() -> int:
        """Get total number of detections (excluding unknowns)"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM detections
                WHERE LOWER(species_en) != 'unknown' AND LOWER(species_nl) != 'onbekend'
            ''')
            return cursor.fetchone()[0]

    @staticmethod
    def get_species_counts() -> dict[str, int]:
        """Get detection count per species (excluding unknowns)"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT species_nl, COUNT(*) as count
                FROM detections
                WHERE LOWER(species_en) != 'unknown' AND LOWER(species_nl) != 'onbekend'
                GROUP BY species_nl
                ORDER BY count DESC
            ''')
            return {row['species_nl']: row['count'] for row in cursor.fetchall()}

    @staticmethod
    def get_top_species(limit: int = 10) -> list[tuple]:
        """Get top N species by detection count (excluding unknowns)"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT species_nl, COUNT(*) as count
                FROM detections
                WHERE LOWER(species_en) != 'unknown' AND LOWER(species_nl) != 'onbekend'
                GROUP BY species_nl
                ORDER BY count DESC
                LIMIT ?
            ''', (limit,))
            return [(row['species_nl'], row['count']) for row in cursor.fetchall()]

    @staticmethod
    def get_recent_detections(limit: int = 10) -> list[dict]:
        """Get most recent detections (excluding unknowns)"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT species_nl as species, camera, confidence, timestamp, image_path, thumbnail_path
                FROM detections
                WHERE LOWER(species_en) != 'unknown' AND LOWER(species_nl) != 'onbekend'
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_daily_counts(days: int = 7) -> dict[str, int]:
        """Get detection counts per day for the last N days (excluding unknowns)"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DATE(timestamp, 'unixepoch') as date, COUNT(*) as count
                FROM detections
                WHERE timestamp >= strftime('%s', 'now', ?)
                  AND LOWER(species_en) != 'unknown' AND LOWER(species_nl) != 'onbekend'
                GROUP BY date
                ORDER BY date
            ''', (f'-{days} days',))
            return {row['date']: row['count'] for row in cursor.fetchall()}

    @staticmethod
    def get_avg_inference_time() -> int | None:
        """Get average inference time in milliseconds"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT AVG(inference_time_ms) FROM detections WHERE inference_time_ms IS NOT NULL')
            result = cursor.fetchone()[0]
            return int(result) if result else None

    @staticmethod
    def get_stats_summary() -> dict[str, Any]:
        """Get complete stats summary (compatible with old JSON format)"""
        return {
            'total_detections': BirdStats.get_total_detections(),
            'species_count': BirdStats.get_species_counts(),
            'last_detections': BirdStats.get_recent_detections(50),
            'daily_counts': BirdStats.get_daily_counts(7),
            'avg_inference_time_ms': BirdStats.get_avg_inference_time()
        }


class ClassificationLog:
    """Manage classification log for all classifications including unknowns"""

    @staticmethod
    def get_entries(limit: int = 50, offset: int = 0, status: str = None,
                    unknowns_only: bool = False) -> list[dict]:
        """Get classification log entries with optional filters"""
        import json
        with get_db() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status)

            if unknowns_only:
                conditions.append("(LOWER(species_en) = 'unknown' OR LOWER(species_nl) = 'onbekend')")

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f'''
                SELECT id, species_en, species_nl, confidence, camera, timestamp,
                       top_predictions, image_path, status, species_corrected,
                       corrected_at, notes, created_at
                FROM classification_log
                {where_clause}
                ORDER BY timestamp DESC, id DESC
                LIMIT ? OFFSET ?
            ''', (*params, limit, offset))

            entries = []
            for row in cursor.fetchall():
                entry = dict(row)
                if entry['top_predictions']:
                    entry['top_predictions'] = json.loads(entry['top_predictions'])
                entries.append(entry)

            return entries

    @staticmethod
    def get_entry(entry_id: int) -> dict | None:
        """Get a single classification log entry by ID"""
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, species_en, species_nl, confidence, camera, timestamp,
                       top_predictions, image_path, status, species_corrected,
                       corrected_at, notes, created_at
                FROM classification_log
                WHERE id = ?
            ''', (entry_id,))
            row = cursor.fetchone()
            if row:
                entry = dict(row)
                if entry['top_predictions']:
                    entry['top_predictions'] = json.loads(entry['top_predictions'])
                return entry
            return None

    @staticmethod
    def update_status(entry_id: int, status: str, species_corrected: str = None,
                      notes: str = None) -> bool:
        """Update the status of a classification log entry"""
        with get_db() as conn:
            cursor = conn.cursor()
            if status == 'corrected' and species_corrected:
                cursor.execute('''
                    UPDATE classification_log
                    SET status = ?, species_corrected = ?, corrected_at = CURRENT_TIMESTAMP, notes = ?
                    WHERE id = ?
                ''', (status, species_corrected, notes, entry_id))
            else:
                cursor.execute('''
                    UPDATE classification_log
                    SET status = ?, notes = ?
                    WHERE id = ?
                ''', (status, notes, entry_id))

            return cursor.rowcount > 0

    @staticmethod
    def get_count(status: str = None, unknowns_only: bool = False) -> int:
        """Get count of classification log entries"""
        with get_db() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if status:
                conditions.append("status = ?")
                params.append(status)

            if unknowns_only:
                conditions.append("(LOWER(species_en) = 'unknown' OR LOWER(species_nl) = 'onbekend')")

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f'SELECT COUNT(*) FROM classification_log {where_clause}', params)
            return cursor.fetchone()[0]

    @staticmethod
    def get_stats() -> dict:
        """Get statistics about classification log"""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM classification_log')
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM classification_log WHERE status = 'pending'")
            pending = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM classification_log WHERE status = 'confirmed'")
            confirmed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM classification_log WHERE status = 'corrected'")
            corrected = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM classification_log
                WHERE LOWER(species_en) = 'unknown' OR LOWER(species_nl) = 'onbekend'
            """)
            unknowns = cursor.fetchone()[0]

            return {
                'total': total,
                'pending': pending,
                'confirmed': confirmed,
                'corrected': corrected,
                'unknowns': unknowns
            }


class BirdConfig:
    """Manage configuration using SQLite"""

    # Default configuration values
    DEFAULTS = {
        'MQTT_BROKER': 'homeassistant.local',
        'MQTT_PORT': '1883',
        'MQTT_USERNAME': '',
        'MQTT_PASSWORD': '',
        'MODEL_NAME': 'birder-project/convnext_v2_tiny_eu-common',
        'MIN_CONFIDENCE': '60',
    }

    @staticmethod
    def get(key: str, default: str = None) -> str | None:
        """Get a config value"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM config WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                return row['value']
            return default or BirdConfig.DEFAULTS.get(key)

    @staticmethod
    def set(key: str, value: str):
        """Set a config value"""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO config (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))

    @staticmethod
    def get_all() -> dict[str, str]:
        """Get all config values"""
        config = dict(BirdConfig.DEFAULTS)
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM config')
            for row in cursor.fetchall():
                config[row['key']] = row['value']
        return config

    @staticmethod
    def set_all(config: dict[str, str]):
        """Set multiple config values"""
        with get_db() as conn:
            cursor = conn.cursor()
            for key, value in config.items():
                if value is not None:
                    cursor.execute('''
                        INSERT OR REPLACE INTO config (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    ''', (key, str(value)))

    @staticmethod
    def load_from_env():
        """Load config from environment variables (migration helper)"""
        import os
        env_keys = ['MQTT_BROKER', 'MQTT_PORT', 'MQTT_USERNAME', 'MQTT_PASSWORD',
                    'MODEL_NAME', 'MIN_CONFIDENCE']
        config = {}
        for key in env_keys:
            value = os.getenv(key)
            if value:
                config[key] = value
        if config:
            BirdConfig.set_all(config)
            logger.info(f"Loaded {len(config)} config values from environment")


# Initialize database on module import
init_db()
