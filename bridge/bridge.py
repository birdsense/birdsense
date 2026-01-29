#!/usr/bin/env python3
"""
BirdSense - Bird Identification API
AI-powered bird species classification with MQTT event publishing
"""

import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
import requests
from classifier import BirdClassifier
from config import Config
from database import BirdStats, get_db
from flasgger import Swagger
from flask import Flask, jsonify, request, send_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Logo path for watermarking
LOGO_PATH = Path(__file__).parent / "logo.png"


def add_watermark(image_path: str, species_name: str, output_path: str) -> bool:
    """
    Add BirdSense watermark to image: logo top-left, species name bottom.

    Args:
        image_path: Path to the original image
        species_name: Bird species name to display at bottom
        output_path: Path to save watermarked image

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        with Image.open(image_path) as img:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            img_width, img_height = img.size

            # Create overlay for watermark
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Add logo (top-left)
            if LOGO_PATH.exists():
                with Image.open(LOGO_PATH) as logo:
                    # Resize logo to ~15% of image width
                    logo_width = int(img_width * 0.15)
                    logo_ratio = logo_width / logo.width
                    logo_height = int(logo.height * logo_ratio)
                    logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

                    # Convert logo to RGBA if needed
                    if logo.mode != 'RGBA':
                        logo = logo.convert('RGBA')

                    # Position: top-left with more padding (5%)
                    padding = int(img_width * 0.05)
                    logo_pos = (padding, padding)

                    # Paste logo with transparency
                    overlay.paste(logo, logo_pos, logo)

            # Add species name (bottom left, aligned with logo) - subtle white text
            font_size = int(img_height * 0.025)  # 2.5% of image height
            font = None
            for font_path in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except OSError:
                    continue
            if font is None:
                font = ImageFont.load_default()

            text = species_name
            bbox = draw.textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]

            # Align left, same padding as logo (5%)
            text_x = int(img_width * 0.05)
            text_y = img_height - text_height - int(img_height * 0.03)

            # Subtle stroke for readability
            stroke_width = max(2, int(font_size * 0.06))
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 220),
                      stroke_width=stroke_width, stroke_fill=(0, 0, 0, 180))

            # Composite overlay onto original
            watermarked = Image.alpha_composite(img, overlay)

            # Convert back to RGB for JPEG saving
            watermarked = watermarked.convert('RGB')
            watermarked.save(output_path, 'JPEG', quality=92)

            return True

    except Exception as e:
        logger.warning(f"Failed to add watermark: {e}")
        return False


class BirdIdentificationBridge:
    """BirdSense: AI bird classifier with MQTT event publishing"""

    def __init__(self):
        self.config = Config()
        self.mqtt_client = None
        self.classifier = None
        self.setup_mqtt()

    def setup_mqtt(self):
        """Initialize MQTT client with authentication"""
        # Use CallbackAPIVersion.VERSION2 to avoid deprecation warning
        self.mqtt_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="bird-bridge"
        )

        if self.config.MQTT_USERNAME:
            self.mqtt_client.username_pw_set(
                self.config.MQTT_USERNAME,
                self.config.MQTT_PASSWORD
            )

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect

    def setup_classifier(self):
        """Initialize the bird classifier"""
        logger.info("Loading bird classification model...")
        logger.info(f"Model: {self.config.MODEL_NAME}")
        logger.info(f"Translations: {self.config.TRANSLATION_FILE}")

        self.classifier = BirdClassifier(
            model_name=self.config.MODEL_NAME,
            translation_file=self.config.TRANSLATION_FILE
        )

        if self.classifier.health_check():
            logger.info("Bird classifier loaded successfully")
        else:
            raise RuntimeError("Failed to load bird classifier")

    def on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback when connected to MQTT broker (MQTT v2 API)"""
        if reason_code == 0:
            logger.info(f"Connected to MQTT broker at {self.config.MQTT_BROKER}")
        else:
            logger.error(f"Failed to connect to MQTT broker: {reason_code}")

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        """Callback when disconnected from MQTT broker (MQTT v2 API)"""
        if reason_code != 0:
            logger.warning(f"Unexpected MQTT disconnect: {reason_code}. Reconnecting...")

    def analyze_bird(self, image_path: str) -> dict[str, Any] | None:
        """
        Classify bird image using EfficientNet model

        Returns:
            Dict with 'species', 'species_en', 'species_nl', 'confidence' or None on error
        """
        if not self.classifier:
            logger.error("Classifier not initialized")
            return None

        result = self.classifier.classify(image_path)

        if result:
            logger.info(f"Classification: {result['species_nl']} ({result['confidence']}%)")
            return result

        return None

    def publish_bird_detection(self, bird_data: dict):
        """Publish bird detection to MQTT"""
        try:
            self.mqtt_client.publish(
                self.config.BIRD_DETECTION_TOPIC,
                json.dumps(bird_data),
                retain=True,
                qos=1
            )
            logger.info(f"Published: {bird_data['species']} on {bird_data['camera']}")
        except Exception as e:
            logger.error(f"Failed to publish bird detection: {e}")

    def run(self, skip_setup=False):
        """Start the bridge service"""
        try:
            if not skip_setup:
                # Validate configuration
                self.config.validate()

                # Initialize classifier (this downloads the model if needed)
                self.setup_classifier()

            logger.info("=" * 60)
            logger.info("BirdSense - Bird Identification API")
            logger.info("=" * 60)
            logger.info(f"Model:          {self.config.MODEL_NAME}")
            logger.info(f"MQTT Broker:    {self.config.MQTT_BROKER}:{self.config.MQTT_PORT}")
            logger.info(f"Min Confidence: {self.config.MIN_CONFIDENCE_THRESHOLD}%")
            logger.info("=" * 60)

            # Connect to MQTT broker with retry
            self._connect_mqtt_with_retry()

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            if self.mqtt_client:
                self.mqtt_client.disconnect()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise

    def _connect_mqtt_with_retry(self, max_retries=5, retry_delay=10):
        """Connect to MQTT broker with retry logic"""
        if not self.config.MQTT_BROKER:
            logger.warning("MQTT_BROKER not configured - running in API-only mode")
            logger.info("API is available at http://localhost:5001")
            logger.info("Configure MQTT_BROKER to enable event publishing")
            # Keep the main thread alive for the API server
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                return

        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to MQTT broker... (attempt {attempt + 1}/{max_retries})")
                self.mqtt_client.connect(
                    self.config.MQTT_BROKER,
                    self.config.MQTT_PORT,
                    60
                )
                logger.info("MQTT connected. API ready for bird classification.")
                self.mqtt_client.loop_start()  # Run MQTT in background thread
                # Keep main thread alive for the API server
                try:
                    while True:
                        time.sleep(60)
                except KeyboardInterrupt:
                    return

            except OSError as e:
                logger.error(f"MQTT connection failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Running in API-only mode.")
                    logger.info("API is still available at http://localhost:5001")
                    logger.info("Fix MQTT_BROKER setting and restart to enable event publishing")
                    # Keep running for the API server
                    try:
                        while True:
                            time.sleep(60)
                    except KeyboardInterrupt:
                        return


# Flask API for web interface
api = Flask(__name__)
api.logger.setLevel(logging.WARNING)  # Reduce Flask logging noise

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger_template = {
    "info": {
        "title": "BirdSense API",
        "description": "AI-powered bird species identification API. Upload an image and receive the predicted bird species with confidence score.",
        "version": "1.0.0",
        "contact": {
            "name": "BirdSense"
        }
    },
    "basePath": "/",
    "schemes": ["http"],
    "tags": [
        {"name": "Classification", "description": "Bird species classification endpoints"},
        {"name": "System", "description": "System and health check endpoints"}
    ]
}

swagger = Swagger(api, config=swagger_config, template=swagger_template)

# Global reference to bridge (set in main)
_bridge_instance = None


@api.route('/api/classify', methods=['POST'])
def api_classify():
    """Classify a bird image
    ---
    tags:
      - Classification
    consumes:
      - multipart/form-data
      - application/x-www-form-urlencoded
    parameters:
      - name: image
        in: formData
        type: file
        required: false
        description: The image to analyze (JPG, PNG). Either image or image_url is required.
      - name: image_url
        in: formData
        type: string
        required: false
        description: URL of the image to analyze. Either image or image_url is required.
      - name: camera
        in: formData
        type: string
        required: false
        description: Camera/source name (optional)
        default: api
      - name: lang
        in: formData
        type: string
        required: false
        description: Language for species names in response (en or nl)
        default: en
        enum: [en, nl]
      - name: skip_stats
        in: formData
        type: boolean
        required: false
        description: Skip saving statistics (for testing)
        default: false
    responses:
      200:
        description: Successful classification
        schema:
          type: object
          properties:
            species:
              type: string
              example: "European Robin (90%)"
              description: Species name with confidence (language depends on lang parameter)
            species_nl:
              type: string
              example: "Roodborst"
              description: Dutch species name
            species_en:
              type: string
              example: "European Robin"
              description: English species name
            confidence:
              type: integer
              example: 90
              description: Confidence score (0-100)
            top_predictions:
              type: array
              items:
                type: object
                properties:
                  species:
                    type: string
                  species_nl:
                    type: string
                  species_en:
                    type: string
                  confidence:
                    type: integer
      400:
        description: No image provided (need either image file or image_url)
      500:
        description: Classification failed
      503:
        description: Classifier not initialized
    """
    global _bridge_instance

    # Ensure classifier is ready
    if _bridge_instance is None or _bridge_instance.classifier is None:
        return jsonify({'error': 'Classifier not initialized'}), 503

    # Get parameters
    camera = request.form.get('camera', 'api')
    lang = request.form.get('lang', 'en').lower()
    if lang not in ('en', 'nl'):
        lang = 'en'
    skip_stats_raw = request.form.get('skip_stats', 'false')
    skip_stats = skip_stats_raw.lower() in ('true', '1', 'yes')

    # Get image from file upload or URL
    tmp_path = None
    image_url = request.form.get('image_url')

    if 'image' in request.files and request.files['image'].filename != '':
        # File upload
        file = request.files['image']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            # Convert to RGB to handle PNG with alpha channel
            from PIL import Image
            img = Image.open(file)
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
            img.save(tmp.name, 'JPEG', quality=95)
            tmp_path = tmp.name
        logger.info(f"API classify: file upload, camera={camera}, lang={lang}, skip_stats={skip_stats}")
    elif image_url:
        # Download from URL
        try:
            logger.info(f"API classify: downloading from URL, camera={camera}, lang={lang}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, timeout=30, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '')
            content = response.content

            # Check if it's an image by content-type OR magic bytes
            # Some CDNs (like BirdBuddy's CloudFront) return wrong content-type
            is_jpeg = content[:3] == b'\xff\xd8\xff'
            is_png = content[:4] == b'\x89PNG'
            is_image_content_type = content_type.startswith('image/')

            if not (is_image_content_type or is_jpeg or is_png):
                logger.warning(f"URL returned non-image: content-type={content_type}, first bytes={content[:20].hex()}")
                return jsonify({'error': f'URL does not point to an image (content-type: {content_type})'}), 400

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            logger.info(f"API classify: image downloaded from URL, camera={camera}")
        except requests.exceptions.Timeout:
            return jsonify({'error': 'Timeout downloading image from URL'}), 400
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Failed to download image: {str(e)}'}), 400
    else:
        return jsonify({'error': 'No image provided. Use either image file upload or image_url parameter.'}), 400

    # Check image size
    image_size = Path(tmp_path).stat().st_size
    max_size = _bridge_instance.config.MAX_IMAGE_SIZE
    if image_size > max_size:
        logger.warning(f"Image too large: {image_size} bytes (max: {max_size})")
        Path(tmp_path).unlink(missing_ok=True)
        return jsonify({'error': f'Image too large ({image_size} bytes, max: {max_size})'}), 400

    # Ensure image directory exists
    image_dir = Path(_bridge_instance.config.IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = _bridge_instance.classifier.classify(tmp_path)
        if result:
            # Save image permanently
            saved_image_path = None
            if _bridge_instance.config.SAVE_IMAGES and result.get('confidence', 0) >= _bridge_instance.config.MIN_CONFIDENCE_THRESHOLD:
                timestamp = int(time.time())
                species_safe = result.get('species_en', 'unknown').replace(' ', '_').lower()
                saved_image_path = str(image_dir / f"{species_safe}_{timestamp}.jpg")

                # Create thumbnail for dashboard
                thumbnail_path = str(image_dir / f"thumb_{species_safe}_{timestamp}.jpg")
                try:
                    from PIL import Image
                    with Image.open(tmp_path) as img:
                        # Calculate thumbnail size (max 300px width, maintain aspect ratio)
                        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                        img.save(thumbnail_path, 'JPEG', quality=85)
                    logger.info(f"Thumbnail created: {thumbnail_path}")
                except Exception as e:
                    logger.warning(f"Failed to create thumbnail: {e}")
                    thumbnail_path = saved_image_path  # Fallback to full image

                # Save image with watermark
                species_display = result.get('species_nl', result.get('species_en', 'Unknown'))
                if add_watermark(tmp_path, species_display, saved_image_path):
                    logger.info(f"Image saved with watermark: {saved_image_path}")
                else:
                    # Fallback: save without watermark
                    shutil.copy2(tmp_path, saved_image_path)
                    logger.info(f"Image saved (no watermark): {saved_image_path}")
            else:
                # Clean up temp file if not saving
                Path(tmp_path).unlink(missing_ok=True)

            # Update species field based on language preference
            confidence = result.get('confidence', 0)
            if lang == 'nl':
                species_name = result.get('species_nl', result.get('species_en', 'Unknown'))
            else:
                species_name = result.get('species_en', 'Unknown')
            result['species'] = f"{species_name} ({confidence}%)"

            # Update top_predictions species field based on language
            for pred in result.get('top_predictions', []):
                if lang == 'nl':
                    pred_name = pred.get('species_nl', pred.get('species_en', 'Unknown'))
                else:
                    pred_name = pred.get('species_en', 'Unknown')
                pred['species'] = pred_name

            # Save to stats and publish MQTT event unless skipped
            if not skip_stats:
                min_conf = _bridge_instance.config.MIN_CONFIDENCE_THRESHOLD
                if confidence >= min_conf:
                    BirdStats.add_detection(
                        species_nl=result.get('species_nl', 'Unknown'),
                        species_en=result.get('species_en'),
                        camera=camera,
                        confidence=confidence,
                        inference_time_ms=result.get('inference_time_ms'),
                        image_path=saved_image_path,
                        thumbnail_path=thumbnail_path if thumbnail_path != saved_image_path else None
                    )
                    # Publish to MQTT if configured
                    if _bridge_instance.config.MQTT_BROKER:
                        bird_data = {
                            'species': result['species'],
                            'species_nl': result.get('species_nl', 'Unknown'),
                            'species_en': result.get('species_en', 'Unknown'),
                            'confidence': confidence,
                            'camera': camera,
                            'timestamp': int(time.time()),
                            'image_path': saved_image_path,
                            'top_predictions': result.get('top_predictions', [])
                        }
                        _bridge_instance.publish_bird_detection(bird_data)
            return jsonify(result)
        else:
            Path(tmp_path).unlink(missing_ok=True)
            return jsonify({'error': 'Classification failed'}), 500
    except Exception as e:
        logger.error(f"Classification error: {e}")
        Path(tmp_path).unlink(missing_ok=True)
        return jsonify({'error': str(e)}), 500


@api.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Service is operational
        schema:
          type: object
          properties:
            status:
              type: string
              example: "ok"
            classifier_ready:
              type: boolean
              example: true
      503:
        description: Service is still initializing
        schema:
          type: object
          properties:
            status:
              type: string
              example: "initializing"
    """
    global _bridge_instance
    if _bridge_instance and _bridge_instance.classifier:
        return jsonify({
            'status': 'ok',
            'classifier_ready': _bridge_instance.classifier.health_check()
        })
    return jsonify({'status': 'initializing'}), 503


@api.route('/images/<path:filename>', methods=['GET'])
def api_serve_image(filename):
    """Serve a saved bird image
    ---
    tags:
      - Images
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Image filename
    responses:
      200:
        description: Image file
        content:
          image/jpeg:
            schema:
              type: string
              format: binary
      400:
        description: Invalid filename
      404:
        description: Image not found
      503:
        description: Bridge not initialized
    """
    global _bridge_instance

    if _bridge_instance is None:
        return jsonify({'error': 'Bridge not initialized'}), 503

    # Security: only allow safe filenames
    safe_filename = Path(filename).name
    if safe_filename != filename:
        return jsonify({'error': 'Invalid filename'}), 400

    image_path = Path(_bridge_instance.config.IMAGE_DIR) / safe_filename

    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    response = send_file(image_path, mimetype='image/jpeg')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    return response


@api.route('/api/images/latest/<int:index>', methods=['GET'])
def api_latest_image(index):
    """Get the Nth latest bird image directly
    ---
    tags:
      - Images
    parameters:
      - name: index
        in: path
        type: integer
        required: true
        description: Image index (0 = most recent, 1 = second most recent, etc.)
    responses:
      200:
        description: Image file
        content:
          image/jpeg:
            schema:
              type: string
              format: binary
      404:
        description: Image not found
      503:
        description: Bridge not initialized
    """
    global _bridge_instance

    if _bridge_instance is None:
        return jsonify({'error': 'Bridge not initialized'}), 503

    image_dir = Path(_bridge_instance.config.IMAGE_DIR)

    if not image_dir.exists():
        return jsonify({'error': 'No images found'}), 404

    # Get sorted list of images (newest first), excluding thumbnails
    # Sort by timestamp in filename (format: species_timestamp.jpg)
    def get_timestamp(f):
        try:
            # Extract timestamp from filename: species_name_1234567890.jpg
            parts = f.stem.rsplit('_', 1)
            if len(parts) == 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        # Fallback to file mtime
        return int(f.stat().st_mtime)

    images = sorted(
        [f for f in image_dir.glob('*.jpg') if not f.name.startswith('thumb_')],
        key=get_timestamp,
        reverse=True
    )

    if index < 0 or index >= len(images):
        return jsonify({'error': f'Image index {index} not found'}), 404

    image_path = images[index]
    response = send_file(image_path, mimetype='image/jpeg')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


@api.route('/api/images', methods=['GET'])
def api_list_images():
    """List saved bird images
    ---
    tags:
      - Images
    parameters:
      - name: limit
        in: query
        type: integer
        required: false
        description: Maximum number of images to return
        default: 50
      - name: offset
        in: query
        type: integer
        required: false
        description: Offset for pagination
        default: 0
    responses:
      200:
        description: List of images
        schema:
          type: object
          properties:
            images:
              type: array
              items:
                type: object
                properties:
                  filename:
                    type: string
                  species_en:
                    type: string
                  species_nl:
                    type: string
                  timestamp:
                    type: integer
                  url:
                    type: string
                  size:
                    type: integer
            total:
              type: integer
            count:
              type: integer
    """
    global _bridge_instance

    if _bridge_instance is None:
        return jsonify({'error': 'Bridge not initialized'}), 503

    image_dir = Path(_bridge_instance.config.IMAGE_DIR)

    if not image_dir.exists():
        return jsonify({'images': [], 'total': 0, 'count': 0})

    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return jsonify({'error': 'Invalid limit/offset'}), 400

    # Get all image files (exclude thumbnails)
    images = []
    for f in image_dir.glob('*.jpg'):
        # Skip thumbnails
        if f.name.startswith('thumb_'):
            continue

        # Parse filename: species_timestamp.jpg
        name = f.stem
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            species_en = parts[0].replace('_', ' ').title()
            try:
                timestamp = int(parts[1])
            except ValueError:
                timestamp = 0
                species_en = name.replace('_', ' ').title()
        else:
            timestamp = 0
            species_en = name.replace('_', ' ').title()

        # Get Dutch species name from database if available
        species_nl = species_en  # Default to English name
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                # Use LOWER() for case-insensitive match
                cursor.execute(
                    'SELECT species_nl FROM detections '
                    'WHERE LOWER(species_en) = LOWER(?) '
                    'ORDER BY timestamp DESC LIMIT 1',
                    (species_en,)
                )
                row = cursor.fetchone()
                if row and row['species_nl']:
                    species_nl = row['species_nl']
        except Exception as e:
            logger.error(f"Error querying Dutch species name: {e}")

        images.append({
            'filename': f.name,
            'species_en': species_en,
            'species_nl': species_nl,
            'timestamp': timestamp,
            'url': f"/images/{f.name}",
            'size': f.stat().st_size
        })

    # Sort by timestamp (newest first)
    images.sort(key=lambda x: x['timestamp'], reverse=True)

    total = len(images)
    paginated = images[offset:offset + limit]

    return jsonify({
        'images': paginated,
        'total': total,
        'count': len(paginated)
    })


@api.route('/api/reload-config', methods=['POST'])
def api_reload_config():
    """Reload configuration without restart
    ---
    tags:
      - System
    responses:
      200:
        description: Configuration reloaded successfully
        schema:
          type: object
          properties:
            status:
              type: string
              example: "ok"
            message:
              type: string
              example: "Configuration reloaded"
            model_changed:
              type: boolean
              example: false
              description: True if the AI model was changed (requires restart)
            note:
              type: string
              example: null
              description: Warning if restart is required
      500:
        description: Error reloading configuration
      503:
        description: Bridge not initialized
    """
    global _bridge_instance
    if _bridge_instance:
        try:
            old_model = _bridge_instance.config.MODEL_NAME
            _bridge_instance.config.reload()
            new_model = _bridge_instance.config.MODEL_NAME

            # Check if model changed (requires classifier reload)
            model_changed = old_model != new_model

            logger.info(f"Configuration reloaded. Min confidence: {_bridge_instance.config.MIN_CONFIDENCE_THRESHOLD}%")

            return jsonify({
                'status': 'ok',
                'message': 'Configuration reloaded',
                'model_changed': model_changed,
                'note': 'Model change requires container restart' if model_changed else None
            })
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'error', 'message': 'Bridge not initialized'}), 503


def run_api_server():
    """Run Flask API server in a separate thread"""
    api.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)


if __name__ == "__main__":
    bridge = BirdIdentificationBridge()
    _bridge_instance = bridge

    # Initialize classifier before starting API server
    bridge.config.validate()
    bridge.setup_classifier()

    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    logger.info("Classifier API server started on port 5001")

    # Run the MQTT bridge (skip_setup=True since we already did it)
    bridge.run(skip_setup=True)
