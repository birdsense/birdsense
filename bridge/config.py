"""
BirdSense Configuration
Loads from SQLite database with .env fallback for migration
"""

import os

from dotenv import load_dotenv

# Path to legacy config file
CONFIG_FILE = '/config/.env'

# Settings that are not editable in the web UI but can be set via environment
ENV_ONLY_KEYS = ('SAVE_IMAGES', 'IMAGE_DIR', 'MAX_IMAGE_SIZE')


def _to_int(value, default: int) -> int:
    """Parse an int config value, falling back to default on empty/invalid input"""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


class Config:
    """Configuration class for BirdSense Bird Identification Bridge"""

    def __init__(self):
        self.reload()

    def reload(self):
        """Reload configuration from database (with .env fallback)"""
        # Import here to avoid circular imports
        from database import BirdConfig

        # Load from .env first (for migration/fallback)
        if os.path.exists(CONFIG_FILE):
            load_dotenv(CONFIG_FILE, override=True)

        # Precedence: values saved via the web UI (database) > environment > defaults.
        # Environment variables only seed settings that were never saved in the UI;
        # otherwise docker-compose's always-set env vars (e.g. MIN_CONFIDENCE) would
        # silently override every change made on the Config page.
        stored = BirdConfig.get_stored()
        config = dict(BirdConfig.DEFAULTS)
        for key in (*config, *ENV_ONLY_KEYS):
            env_val = os.getenv(key)
            if env_val:
                config[key] = env_val
        config.update(stored)

        # MQTT settings
        self.MQTT_BROKER = config.get('MQTT_BROKER', 'homeassistant.local')
        self.MQTT_PORT = _to_int(config.get('MQTT_PORT'), 1883)
        self.MQTT_USERNAME = config.get('MQTT_USERNAME') or None
        self.MQTT_PASSWORD = config.get('MQTT_PASSWORD') or None

        # Model settings
        self.MODEL_NAME = config.get('MODEL_NAME', 'birder-project/convnext_v2_tiny_eu-common')
        self.TRANSLATION_FILE = os.getenv('TRANSLATION_FILE', '/app/bird_names_nl.json')

        # MQTT topic for publishing bird detections
        self.BIRD_DETECTION_TOPIC = 'birdsense/detections'

        # Identification settings
        self.MIN_CONFIDENCE_THRESHOLD = _to_int(config.get('MIN_CONFIDENCE'), 60)

        # Image storage settings
        self.SAVE_IMAGES = str(config.get('SAVE_IMAGES', 'true')).lower() in ('true', '1', 'yes')
        self.IMAGE_DIR = config.get('IMAGE_DIR', '/data/images')
        self.MAX_IMAGE_SIZE = _to_int(config.get('MAX_IMAGE_SIZE'), 10485760)  # 10MB default

    def validate(self):
        """Validate required configuration"""
        required = [
            ('MODEL_NAME', self.MODEL_NAME),
        ]

        # MQTT is optional (for event publishing)
        missing = [name for name, value in required if not value]

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
