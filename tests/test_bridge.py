"""Smoke tests for the bridge API and config (classifier and torch are stubbed)."""

import io
import sys
import threading
import time
import types

from conftest import make_jpeg_bytes


def test_config_database_overrides_env(import_component, monkeypatch):
    monkeypatch.setenv('MIN_CONFIDENCE', '30')
    database = import_component('bridge', 'database')
    config = import_component('bridge', 'config')

    # Env seeds settings that were never saved in the web UI
    assert config.Config().MIN_CONFIDENCE_THRESHOLD == 30

    # A value saved via the web UI wins over the always-set docker-compose env var
    database.BirdConfig.set('MIN_CONFIDENCE', '75')
    assert config.Config().MIN_CONFIDENCE_THRESHOLD == 75


def test_config_invalid_numbers_fall_back_to_defaults(import_component):
    database = import_component('bridge', 'database')
    config = import_component('bridge', 'config')
    database.BirdConfig.set_all({'MQTT_PORT': '', 'MIN_CONFIDENCE': 'abc'})

    cfg = config.Config()
    assert cfg.MQTT_PORT == 1883
    assert cfg.MIN_CONFIDENCE_THRESHOLD == 60


def test_web_and_bridge_share_schema(import_component):
    bridge_db = import_component('bridge', 'database')
    tables = {}
    with bridge_db.get_db() as conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            tables[name] = [row[1] for row in conn.execute(f'PRAGMA table_info({name})')]

    sys.modules.pop('database', None)
    web_db = import_component('web', 'database')
    web_db.init_db()  # must be a no-op on a schema created by the bridge
    with web_db.get_db() as conn:
        for name, columns in tables.items():
            assert [row[1] for row in conn.execute(f'PRAGMA table_info({name})')] == columns
        assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'


def test_health(bridge_module):
    response = bridge_module.api.test_client().get('/api/health')
    assert response.status_code == 200
    assert response.get_json()['classifier_ready'] is True


def test_classify_upload_saves_unique_images(bridge_module, data_dir):
    client = bridge_module.api.test_client()
    for _ in range(2):
        response = client.post('/api/classify', data={
            'image': (io.BytesIO(make_jpeg_bytes()), 'bird.jpg'),
            'camera': 'test',
        }, content_type='multipart/form-data')
        assert response.status_code == 200
        assert response.get_json()['species'] == 'European Robin (90%)'

    saved = sorted(p.name for p in (data_dir / 'images').iterdir() if not p.name.startswith('thumb_'))
    # Same species within the same second must not overwrite the previous image
    assert len(saved) == 2

    log = client.get('/api/log').get_json()
    assert log['total'] == 2


def test_classify_rejects_invalid_upload(bridge_module):
    client = bridge_module.api.test_client()
    response = client.post('/api/classify', data={
        'image': (io.BytesIO(b'definitely not an image'), 'bird.jpg'),
    }, content_type='multipart/form-data')
    assert response.status_code == 400

    assert client.post('/api/classify', data={}).status_code == 400


def test_serve_image_rejects_traversal(bridge_module, data_dir):
    (data_dir / 'images' / 'bird.jpg').write_bytes(make_jpeg_bytes())
    client = bridge_module.api.test_client()
    assert client.get('/images/bird.jpg').status_code == 200
    assert client.get('/images/missing.jpg').status_code == 404
    assert client.get('/images/..%2fbirdsense.db').status_code in (400, 404)


def test_update_log_triggers_auto_train(bridge_module, monkeypatch):
    calls = []
    fake_train = types.ModuleType('train')
    fake_train.get_training_status = lambda: types.SimpleNamespace(running=False)
    fake_train.start_training = lambda **kwargs: calls.append(kwargs) or True
    monkeypatch.setitem(sys.modules, 'train', fake_train)

    database = sys.modules['database']
    ids = [
        database.ClassificationLog.add_entry('European Robin', 'Roodborst', 80, 'cam', timestamp=i)
        for i in range(6)
    ]
    client = bridge_module.api.test_client()

    # Fewer than 5 labeled samples: no training
    for entry_id in ids[:4]:
        response = client.put(f'/api/log/{entry_id}', json={'status': 'confirmed'})
        assert response.get_json()['training'] == 'skipped'
    assert calls == []

    response = client.put(f'/api/log/{ids[4]}', json={'status': 'corrected', 'species_corrected': 'Great Tit'})
    assert response.get_json()['training'] == 'started'
    assert len(calls) == 1
    assert calls[0]['classifier'] is bridge_module._bridge_instance.classifier

    assert client.put('/api/log/99999', json={'status': 'confirmed'}).status_code == 404
    assert client.put(f'/api/log/{ids[5]}', data='x').status_code == 400


def test_start_training_validates_params(bridge_module, monkeypatch):
    started = []
    fake_train = types.ModuleType('train')
    fake_train.get_training_status = lambda: types.SimpleNamespace(to_dict=lambda: {'running': True})
    fake_train.start_training = lambda **kwargs: started.append(kwargs) or len(started) == 1
    monkeypatch.setitem(sys.modules, 'train', fake_train)

    client = bridge_module.api.test_client()
    assert client.post('/api/train', json={'epochs': 'ten'}).status_code == 400
    assert client.post('/api/train', json={'epochs': 0}).status_code == 400

    assert client.post('/api/train', json={'epochs': '3', 'freeze_backbone': 'false'}).status_code == 200
    assert started[0]['epochs'] == 3 and started[0]['freeze_backbone'] is False
    # A second request while the first run is active is rejected
    assert client.post('/api/train', json={}).status_code == 409


def _stub_torch(monkeypatch):
    """Minimal torch/torchvision stand-ins so train.py can be imported without torch."""
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    utils = types.ModuleType('torch.utils')
    data = types.ModuleType('torch.utils.data')
    data.Dataset = object
    data.DataLoader = data.random_split = None
    torchvision = types.ModuleType('torchvision')
    torchvision.transforms = types.ModuleType('torchvision.transforms')
    stubs = {
        'torch': torch,
        'torch.utils': utils,
        'torch.utils.data': data,
        'torchvision': torchvision,
        'torchvision.transforms': torchvision.transforms,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_start_training_allows_only_one_run(import_component, monkeypatch):
    _stub_torch(monkeypatch)
    train = import_component('bridge', 'train')
    release = threading.Event()
    runs = []

    def fake_run(**kwargs):
        runs.append(kwargs)
        assert kwargs['_status_ready'] is True
        release.wait(5)
        train.training_status.running = False

    monkeypatch.setattr(train, 'run_training', fake_run)
    classifier = object()

    assert train.start_training(classifier=classifier, epochs=3) is True
    # Status flips to running synchronously, before the thread does any work
    assert train.get_training_status().running is True
    assert train.get_training_status().total_epochs == 3
    assert train.start_training(classifier=classifier, epochs=3) is False

    release.set()
    for _ in range(100):
        if not train.get_training_status().running:
            break
        time.sleep(0.01)
    assert len(runs) == 1 and runs[0]['classifier'] is classifier
