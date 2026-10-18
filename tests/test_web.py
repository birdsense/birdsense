"""Smoke tests for the web dashboard (web/app.py)."""

import pytest


@pytest.mark.parametrize('path', ['/', '/config', '/test', '/log', '/training', '/api/stats'])
def test_pages_render(web_client, path):
    response = web_client.get(path)
    assert response.status_code == 200


@pytest.mark.parametrize('query', ['offset=abc', 'offset=-20', 'filter=unknowns', 'filter=pending&offset=20'])
def test_log_handles_query_params(web_client, query):
    assert web_client.get(f'/log?{query}').status_code == 200


def test_dashboard_shows_detection(web_client, import_component):
    database = import_component('web', 'database')
    database.BirdStats.add_detection(species_nl='Roodborst', species_en='European Robin', camera='cam', confidence=91)
    response = web_client.get('/')
    assert response.status_code == 200
    assert 'Roodborst' in response.get_data(as_text=True)


def test_image_serving_blocks_traversal(web_client, data_dir):
    (data_dir / 'images' / 'bird.jpg').write_bytes(b'jpegdata')
    (data_dir / 'secret.txt').write_text('secret')

    assert web_client.get('/images/bird.jpg').status_code == 200
    assert web_client.get('/images/../secret.txt').status_code == 404
    assert web_client.get('/images/%2e%2e/secret.txt').status_code == 404
    assert web_client.get('/images/missing.jpg').status_code == 404


def test_config_post_rejects_invalid_numbers(web_client, import_component):
    response = web_client.post('/config', data={
        'mqtt_broker': ' broker.local ',
        'mqtt_port': '',
        'mqtt_username': '',
        'mqtt_password': '',
        'model_name': 'birder-project/convnext_v2_tiny_eu-common',
        'min_confidence': '150',
    })
    assert response.status_code == 200

    database = import_component('web', 'database')
    stored = database.BirdConfig.get_all()
    assert stored['MQTT_BROKER'] == 'broker.local'
    assert stored['MQTT_PORT'] == '1883'
    assert stored['MIN_CONFIDENCE'] == '100'


def test_update_log_validation(web_client):
    assert web_client.put('/api/log/1', data='not json').status_code == 400
    assert web_client.put('/api/log/1', json={'status': 'bogus'}).status_code == 400
    assert web_client.put('/api/log/1', json={'status': 'corrected'}).status_code == 400


def test_update_log_falls_back_to_local_db_when_bridge_down(web_client, import_component):
    database = import_component('web', 'database')
    with database.get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO classification_log (species_en, species_nl, confidence, camera, timestamp) "
            "VALUES ('European Robin', 'Roodborst', 80, 'cam', 1)"
        )
        entry_id = cursor.lastrowid

    response = web_client.put(f'/api/log/{entry_id}', json={'status': 'corrected', 'species_corrected': 'Great Tit'})
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'
    assert database.ClassificationLog.get_entry(entry_id)['species_corrected'] == 'Great Tit'

    assert web_client.put('/api/log/99999', json={'status': 'confirmed'}).status_code == 404


def test_bridge_proxies_degrade_gracefully(web_client):
    assert web_client.get('/api/species').get_json() == {'species': [], 'count': 0}
    assert web_client.get('/api/train/status').get_json()['running'] is False
    assert web_client.post('/api/train', json={}).status_code == 503
