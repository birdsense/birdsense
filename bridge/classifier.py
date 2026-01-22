#!/usr/bin/env python3
"""
BirdSense - Bird Species Classifier
Uses birder-project EU-common model for European bird species classification
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Model configurations
BIRDER_MODELS = {
    "birder-project/convnext_v2_tiny_eu-common": {
        "name": "ConvNeXt V2 EU-Common",
        "species_count": 707,
        "input_size": 384,
        "description": "European birds (Collins Bird Guide)"
    },
    "dennisjooo/Birds-Classifier-EfficientNetB2": {
        "name": "EfficientNetB2",
        "species_count": 525,
        "input_size": 224,
        "description": "Global birds (Kaggle dataset)"
    }
}


class BirdClassifier:
    """
    Bird species classifier supporting multiple backends.

    Supports:
    - birder-project models (European birds)
    - HuggingFace transformers models (global birds)
    """

    def __init__(
        self,
        model_name: str = "birder-project/convnext_v2_tiny_eu-common",
        translation_file: str | None = None,
        device: str | None = None
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_birder_model = model_name.startswith("birder-project/")

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")

        if self.is_birder_model:
            self._load_birder_model(model_name)
        else:
            self._load_transformers_model(model_name)

        # Load translations
        self.translations = {}
        if translation_file and Path(translation_file).exists():
            self._load_translations(translation_file)
            logger.info(f"Loaded {len(self.translations)} Dutch translations")
        else:
            logger.warning("No translation file loaded - will use English names")

        logger.info(f"Model loaded with {len(self.id2label)} species")

    def _load_birder_model(self, model_name: str) -> None:
        """Load a birder-project model."""
        try:
            import birder
            from birder.inference.classification import infer_image

            birder_model_id = model_name.replace("birder-project/", "")

            self.model, self.model_info = birder.load_pretrained_model(birder_model_id, inference=True)
            self.model.to(self.device)

            size = birder.get_size_from_signature(self.model_info.signature)
            self.transform = birder.classification_transform(size, self.model_info.rgb_stats)

            self.id2label = dict(enumerate(self.model_info.class_to_idx.keys()))
            self.infer_func = infer_image

            self.processor = None

        except ImportError as err:
            raise ImportError("birder library not installed. Run: pip install birder") from err

    def _load_transformers_model(self, model_name: str) -> None:
        """Load a HuggingFace transformers model."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.transform = None
        self.infer_func = None

    def _load_translations(self, translation_file: str) -> None:
        try:
            with open(translation_file, encoding='utf-8') as f:
                raw_translations = json.load(f)
            self.translations = {k.lower(): v for k, v in raw_translations.items()}
        except Exception as e:
            logger.error(f"Failed to load translations: {e}")
            self.translations = {}

    def _translate(self, species_en: str) -> str:
        species_lower = species_en.lower()
        return self.translations.get(species_lower, f"{species_en} (EN)")

    def classify(self, image_path: str) -> dict[str, Any] | None:
        try:
            start_time = time.time()
            if self.is_birder_model:
                result = self._classify_birder(image_path)
            else:
                result = self._classify_transformers(image_path)

            if result:
                inference_ms = int((time.time() - start_time) * 1000)
                result['inference_time_ms'] = inference_ms
            return result
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None

    def _classify_birder(self, image_path: str) -> dict[str, Any]:
        """Classify using birder model."""
        from birder.inference.classification import infer_image

        probabilities, _ = infer_image(self.model, image_path, self.transform)
        # probabilities is shape (1, num_classes), squeeze to 1D
        probabilities = probabilities.squeeze(0)

        top_idx = int(np.argmax(probabilities))
        top_prob = float(probabilities[top_idx])
        confidence = int(top_prob * 100)
        species_en = self.id2label[top_idx]

        species_en_clean = species_en.replace('_', ' ').title()
        species_nl = self._translate(species_en_clean)

        top_k = min(3, len(self.id2label))
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            prob = probabilities[idx]
            name = self.id2label[idx].replace('_', ' ').title()
            top_predictions.append({
                'species_en': name,
                'species_nl': self._translate(name),
                'confidence': int(prob * 100)
            })

        result = {
            'species_en': species_en_clean,
            'species_nl': species_nl,
            'species': f"{species_nl} ({confidence}%)",
            'confidence': confidence,
            'top_predictions': top_predictions
        }

        logger.info(f"Classification: {species_nl} ({confidence}%) [EN: {species_en_clean}]")
        return result

    def _classify_transformers(self, image_path: str) -> dict[str, Any]:
        """Classify using HuggingFace transformers model."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        top_prob, top_idx = torch.max(probabilities, dim=-1)
        confidence = int(top_prob.item() * 100)
        species_en = self.id2label[top_idx.item()]

        species_en_clean = species_en.replace('_', ' ').title()
        species_nl = self._translate(species_en_clean)

        top_k = min(3, len(self.id2label))
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0], strict=True):
            name = self.id2label[idx.item()].replace('_', ' ').title()
            top_predictions.append({
                'species_en': name,
                'species_nl': self._translate(name),
                'confidence': int(prob.item() * 100)
            })

        result = {
            'species_en': species_en_clean,
            'species_nl': species_nl,
            'species': f"{species_nl} ({confidence}%)",
            'confidence': confidence,
            'top_predictions': top_predictions
        }

        logger.info(f"Classification: {species_nl} ({confidence}%) [EN: {species_en_clean}]")
        return result

    def health_check(self) -> bool:
        """Check if the model is loaded and ready."""
        try:
            return self.model is not None
        except Exception:
            return False


# Simple test when run directly
if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    translation_file = os.path.join(script_dir, "bird_names_nl.json")

    model = os.environ.get("MODEL_NAME", "birder-project/convnext_v2_tiny_eu-common")
    classifier = BirdClassifier(model_name=model, translation_file=translation_file)

    if len(sys.argv) > 1:
        result = classifier.classify(sys.argv[1])
        if result:
            print(f"\nResult: {result['species']}")
            print(f"English: {result['species_en']}")
            print(f"Confidence: {result['confidence']}%")
            print("\nTop 3 predictions:")
            for pred in result['top_predictions']:
                print(f"  - {pred['species_nl']} ({pred['confidence']}%)")
    else:
        print("Usage: python classifier.py <image_path>")
        print(f"Model loaded: {classifier.health_check()}")
        print(f"Species count: {len(classifier.id2label)}")
