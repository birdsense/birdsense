"""
Shared pytest fixtures.

bridge/ and web/ are separate containers that each have a top-level ``database``
module, so they can't both be on sys.path at once. These fixtures import one
component at a time against a temporary SQLite database and image directory,
and clear the imported modules afterwards.
"""

import importlib
import io
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPONENT_MODULES = ('database', 'config', 'app', 'bridge', 'classifier', 'train')


def _clear_modules():
    for name in COMPONENT_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point both components at a temporary database and image directory."""
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    monkeypatch.setenv('DATABASE_FILE', str(tmp_path / 'birdsense.db'))
    monkeypatch.setenv('IMAGE_DIR', str(image_dir))
    # Unreachable bridge: connection refused immediately
    monkeypatch.setenv('BRIDGE_URL', 'http://127.0.0.1:9')
    for key in ('MQTT_BROKER', 'MQTT_PORT', 'MODEL_NAME', 'MIN_CONFIDENCE'):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def import_component(data_dir, monkeypatch):
    """Return a function that imports a module from bridge/ or web/ in isolation."""
    _clear_modules()

    def _import(component, module):
        monkeypatch.syspath_prepend(str(ROOT / component))
        return importlib.import_module(module)

    yield _import
    _clear_modules()


@pytest.fixture
def web_client(import_component):
    app_module = import_component('web', 'app')
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


class FakeClassifier:
    """Stand-in for BirdClassifier so the bridge API can be tested without torch."""

    def __init__(self, species_en='European Robin', species_nl='Roodborst', confidence=90):
        self.result = {'species_en': species_en, 'species_nl': species_nl, 'confidence': confidence}
        self.reloads = 0

    def classify(self, image_path):
        assert Path(image_path).exists()
        return {
            **self.result,
            'top_predictions': [{**self.result}],
            'inference_time_ms': 5,
        }

    def health_check(self):
        return True

    def reload_model(self):
        self.reloads += 1
        return True


@pytest.fixture
def bridge_module(import_component, monkeypatch):
    """Import bridge.py with the classifier stubbed out and an initialized bridge instance."""
    fake_classifier_module = types.ModuleType('classifier')
    fake_classifier_module.BirdClassifier = FakeClassifier
    monkeypatch.setitem(sys.modules, 'classifier', fake_classifier_module)

    bridge = import_component('bridge', 'bridge')
    instance = bridge.BirdIdentificationBridge()
    instance.config.MQTT_BROKER = ''
    instance.classifier = FakeClassifier()
    monkeypatch.setattr(bridge, '_bridge_instance', instance)
    bridge.api.config['TESTING'] = True
    return bridge


def make_jpeg_bytes(size=(64, 48), color=(200, 30, 30)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'JPEG')
    return buf.getvalue()
