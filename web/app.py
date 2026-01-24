#!/usr/bin/env python3
"""
BirdSense Web Interface
Configuration and dashboard for BirdSense bird identification system
"""

import os
import tempfile
from datetime import datetime, timedelta

import requests
from database import BirdConfig, BirdStats, init_db
from flask import Flask, jsonify, redirect, render_template, request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize database
init_db()

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from shared data directory"""
    try:
        from flask import send_from_directory
        return send_from_directory('/data/images', filename)
    except FileNotFoundError:
        return "Image not found", 404


# Jinja2 filter for timestamp formatting
@app.template_filter('timestamp_to_time')
def timestamp_to_time(timestamp):
    """Convert Unix timestamp to readable time"""
    if timestamp:
        return datetime.fromtimestamp(timestamp).strftime('%H:%M')
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
        config_data = {
            'MQTT_BROKER': request.form.get('mqtt_broker'),
            'MQTT_PORT': request.form.get('mqtt_port'),
            'MQTT_USERNAME': request.form.get('mqtt_username'),
            'MQTT_PASSWORD': request.form.get('mqtt_password'),
            'MODEL_NAME': request.form.get('model_name'),
            'MIN_CONFIDENCE': request.form.get('min_confidence')
        }

        # Save to database
        BirdConfig.set_all(config_data)
        saved = True

        # Try to hot-reload the bridge config
        try:
            bridge_url = os.environ.get('BRIDGE_URL', 'http://host.docker.internal:5001')
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
        bridge_url = os.environ.get('BRIDGE_URL', 'http://host.docker.internal:5001') + '/api/classify'

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
