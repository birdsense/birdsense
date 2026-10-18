#!/usr/bin/env python3
"""
BirdSense Web Interface
Configuration and dashboard for BirdSense bird identification system
"""

import os
import tempfile
from datetime import datetime, timedelta

import requests
from database import BirdConfig, BirdStats, ClassificationLog, init_db
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

IMAGE_DIR = os.environ.get('IMAGE_DIR', '/data/images')


def get_bridge_url() -> str:
    """Internal URL of the bridge API (container-to-container)"""
    return os.environ.get('BRIDGE_URL', 'http://host.docker.internal:5001').rstrip('/')


def _parse_int(value, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse an int from user input, clamping to [minimum, maximum]; default on invalid input"""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


# Initialize database
init_db()

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from shared data directory"""
    # send_from_directory rejects path traversal and returns 404 for missing files
    return send_from_directory(IMAGE_DIR, filename)


# Jinja2 filter for timestamp formatting
@app.template_filter('timestamp_to_time')
def timestamp_to_time(timestamp):
    """Convert Unix timestamp to readable time"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%H:%M')
    return ''


@app.template_filter('format_duration')
def format_duration(ms):
    """Format milliseconds as human-readable duration"""
    if not ms:
        return '-'
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime(timestamp):
    """Convert Unix timestamp to readable date and time"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
    return ''


@app.route('/')
def index():
    """Dashboard homepage"""
    # Get stats from database
    total = BirdStats.get_total_detections()
    species_count = BirdStats.get_species_counts()
    top_species = BirdStats.get_top_species(10)
    recent = BirdStats.get_recent_detections(10)
    daily_counts_dict = BirdStats.get_daily_counts(7)
    avg_inference_time = BirdStats.get_avg_inference_time()

    # Get config from database
    config = BirdConfig.get_all()

    # Format daily counts for chart (ensure all 7 days present)
    last_7_days = []
    for i in range(6, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        count = daily_counts_dict.get(date, 0)
        last_7_days.append({'date': date, 'count': count})

    return render_template('dashboard.html',
                         total=total,
                         species_count=len(species_count),
                         top_species=top_species,
                         recent=recent,
                         daily_counts=last_7_days,
                         avg_inference_time_ms=avg_inference_time,
                         config=config,
                         bridge_url=os.environ.get('BRIDGE_EXTERNAL_URL', 'http://localhost:5001'))


@app.route('/config', methods=['GET', 'POST'])
def config():
    """Configuration page"""
    saved = False
    reload_result = None
    if request.method == 'POST':
        current = BirdConfig.get_all()
        # Numeric fields: fall back to the current value on empty/invalid input, so a
        # blank port can't be saved and later crash the bridge on int() parsing
        mqtt_port = _parse_int(request.form.get('mqtt_port'), _parse_int(current.get('MQTT_PORT'), 1883),
                               minimum=1, maximum=65535)
        min_confidence = _parse_int(request.form.get('min_confidence'),
                                    _parse_int(current.get('MIN_CONFIDENCE'), 60), minimum=0, maximum=100)
        config_data = {
            'MQTT_BROKER': (request.form.get('mqtt_broker') or '').strip(),
            'MQTT_PORT': str(mqtt_port),
            'MQTT_USERNAME': request.form.get('mqtt_username'),
            'MQTT_PASSWORD': request.form.get('mqtt_password'),
            'MODEL_NAME': request.form.get('model_name') or current.get('MODEL_NAME'),
            'MIN_CONFIDENCE': str(min_confidence),
        }

        # Save to database
        BirdConfig.set_all(config_data)
        saved = True

        # Try to hot-reload the bridge config
        try:
            bridge_url = get_bridge_url()
            response = requests.post(f"{bridge_url}/api/reload-config", timeout=5)
            if response.status_code == 200:
                reload_result = response.json()
        except Exception:
            reload_result = None  # Bridge might not be running

    current_config = BirdConfig.get_all()
    return render_template('config.html', config=current_config, saved=saved, reload_result=reload_result)


@app.route('/test')
def test():
    """Test classifier page"""
    return render_template('test.html')


@app.route('/log')
def log():
    """Classification log page"""
    filter_type = request.args.get('filter')
    offset = _parse_int(request.args.get('offset'), 0, minimum=0)
    limit = 20

    # Get filter parameters
    status = None
    unknowns_only = False

    if filter_type == 'unknowns':
        unknowns_only = True
    elif filter_type in ('pending', 'confirmed', 'corrected'):
        status = filter_type

    # Get entries and stats
    entries = ClassificationLog.get_entries(limit=limit, offset=offset, status=status,
                                            unknowns_only=unknowns_only)
    total = ClassificationLog.get_count(status=status, unknowns_only=unknowns_only)
    stats = ClassificationLog.get_stats()

    return render_template('log.html',
                          entries=entries,
                          total=total,
                          stats=stats,
                          filter=filter_type,
                          offset=offset,
                          limit=limit)


@app.route('/api/log/<int:entry_id>', methods=['PUT'])
def api_update_log(entry_id):
    """Update a classification log entry"""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    status = data.get('status')
    if status not in ('confirmed', 'corrected'):
        return jsonify({'error': 'Invalid status'}), 400

    species_corrected = data.get('species_corrected')
    if status == 'corrected' and not species_corrected:
        return jsonify({'error': 'species_corrected is required'}), 400

    notes = data.get('notes')

    # Route the update through the bridge: it writes the same database and also
    # triggers auto-training. Updating the DB directly here would skip training.
    try:
        response = requests.put(
            f"{get_bridge_url()}/api/log/{entry_id}",
            json={'status': status, 'species_corrected': species_corrected, 'notes': notes},
            timeout=10,
        )
        return jsonify(response.json()), response.status_code
    except (requests.exceptions.RequestException, ValueError):
        pass  # Bridge unavailable or returned non-JSON: fall back to a local update

    if ClassificationLog.update_status(entry_id, status, species_corrected, notes):
        return jsonify({'status': 'ok', 'training': 'unavailable'})
    return jsonify({'error': 'Entry not found'}), 404


@app.route('/api/species')
def api_species():
    """Proxy endpoint to get species list from bridge"""
    try:
        bridge_url = get_bridge_url()
        response = requests.get(f"{bridge_url}/api/species", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        return jsonify({'species': [], 'count': 0})
    except Exception:
        return jsonify({'species': [], 'count': 0})


@app.route('/training')
def training():
    """Training page"""
    stats = ClassificationLog.get_stats()
    return render_template('training.html', log_stats=stats)


@app.route('/api/train', methods=['POST'])
def api_train():
    """Proxy training request to bridge"""
    try:
        bridge_url = get_bridge_url()
        response = requests.post(
            f"{bridge_url}/api/train",
            json=request.get_json(silent=True) or {},
            timeout=10,
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': f'Bridge unavailable: {e}'}), 503


@app.route('/api/train/status')
def api_train_status():
    """Proxy training status from bridge"""
    try:
        bridge_url = get_bridge_url()
        response = requests.get(f"{bridge_url}/api/train/status", timeout=5)
        return jsonify(response.json())
    except Exception:
        return jsonify({'running': False, 'error': 'Bridge unavailable'})


@app.route('/api/train/reload', methods=['POST'])
def api_train_reload():
    """Proxy model reload request to bridge"""
    try:
        bridge_url = get_bridge_url()
        response = requests.post(f"{bridge_url}/api/train/reload", timeout=30)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': f'Bridge unavailable: {e}'}), 503


@app.route('/api-docs')
def api_docs():
    """Redirect to Swagger API documentation on bridge"""
    # Use request host to build the URL dynamically
    # This works because both services are typically on the same host, different ports
    host = request.host.split(':')[0]  # Get hostname without port
    bridge_port = os.environ.get('BRIDGE_EXTERNAL_PORT', '5001')
    return redirect(f"http://{host}:{bridge_port}/docs")


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint to classify an uploaded image (stats handled by bridge)"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get optional parameters
    camera = request.form.get('camera', 'web')
    skip_stats = request.form.get('skip_stats', 'false')

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Call the bridge classifier API
        bridge_url = get_bridge_url() + '/api/classify'

        with open(tmp_path, 'rb') as f:
            response = requests.post(
                bridge_url,
                files={'image': f},
                data={'camera': camera, 'skip_stats': skip_stats},
                timeout=30
            )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text
            return jsonify({
                'error': 'Classifier service unavailable',
                'status_code': response.status_code,
                'details': error_detail
            }), 503

    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Could not connect to classifier service. '
                     'Make sure the bridge container is running.'
        }), 503
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify(BirdStats.get_stats_summary())


if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('/data', exist_ok=True)

    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=8080, debug=debug_mode)
