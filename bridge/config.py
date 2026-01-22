"""
BirdSense Configuration
Loads from SQLite database with .env fallback for migration
"""

import os

from dotenv import load_dotenv

# Path to legacy config file
CONFIG_FILE = '/config/.env'


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

        # Get config from database (falls back to env vars and defaults)
        config = BirdConfig.get_all()

        # Also check environment variables (they take precedence)
        for key in config.keys():
            env_val = os.getenv(key)
            if env_val:
                config[key] = env_val

        # MQTT settings
        self.MQTT_BROKER = config.get('MQTT_BROKER', 'homeassistant.local')
        self.MQTT_PORT = int(config.get('MQTT_PORT', 1883))
        self.MQTT_USERNAME = config.get('MQTT_USERNAME') or None
        self.MQTT_PASSWORD = config.get('MQTT_PASSWORD') or None

        # Model settings
        self.MODEL_NAME = config.get('MODEL_NAME', 'birder-project/convnext_v2_tiny_eu-common')
        self.TRANSLATION_FILE = os.getenv('TRANSLATION_FILE', '/app/bird_names_nl.json')

        # MQTT topic for publishing bird detections
        self.BIRD_DETECTION_TOPIC = 'birdsense/detections'

        # Identification settings
        self.MIN_CONFIDENCE_THRESHOLD = int(config.get('MIN_CONFIDENCE', 60))

        # Image storage settings
        self.SAVE_IMAGES = config.get('SAVE_IMAGES', 'true').lower() in ('true', '1', 'yes')
        self.IMAGE_DIR = config.get('IMAGE_DIR', '/data/images')
        self.MAX_IMAGE_SIZE = int(config.get('MAX_IMAGE_SIZE', 10485760))  # 10MB default

    def validate(self):
        """Validate required configuration"""
        required = [
            ('MODEL_NAME', self.MODEL_NAME),
        ]

        # MQTT is optional (for event publishing)
        missing = [name for name, value in required if not value]

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
