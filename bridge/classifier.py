#!/usr/bin/env python3
"""
BirdSense - Bird Species Classifier
Uses birder-project EU-common models for European bird species classification
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

FINETUNED_WEIGHTS = "/data/models/finetuned_latest.pt"
FINETUNED_META = "/data/models/finetuned_meta.json"

# Default model. All "eu-common" birder models share the exact same 707-class
# label set (same order), so Dutch names and logged species labels stay valid
# when switching between them. Fine-tuned weights are tied to one base model.
DEFAULT_MODEL = "birder-project/vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-eu-common"
LEGACY_MODEL = "birder-project/convnext_v2_tiny_eu-common"

# Model configurations. Accuracy / macro-F1 are on the eu-common validation set
# (19.9k images) from the Birder leaderboard
# (https://huggingface.co/spaces/birder-project/leaderboard).
BIRDER_MODELS = {
    "birder-project/vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-eu-common": {
        "name": "ViT Parallel S16 (data2vec, intermediate) EU-Common",
        "species_count": 707,
        "input_size": 384,
        "params_m": 64.6,
        "accuracy": 98.03,
        "description": "European birds (Collins Bird Guide) - recommended, best accuracy/cost balance",
    },
    "birder-project/hiera_abswin_base_mim-intermediate-eu-common": {
        "name": "Hiera AbsWin Base (MIM, intermediate) EU-Common",
        "species_count": 707,
        "input_size": 384,
        "params_m": 51.0,
        "accuracy": 97.80,
        "description": "European birds (Collins Bird Guide) - lighter than ViT Parallel",
    },
    "birder-project/convnext_v2_tiny_intermediate-eu-common": {
        "name": "ConvNeXt V2 Tiny (intermediate) EU-Common",
        "species_count": 707,
        "input_size": 384,
        "params_m": 28.4,
        "accuracy": 97.43,
        "description": "European birds (Collins Bird Guide) - fastest, for low-end hardware",
    },
    "birder-project/convnext_v2_tiny_eu-common": {
        "name": "ConvNeXt V2 Tiny EU-Common (legacy)",
        "species_count": 707,
        "input_size": 384,
        "params_m": 28.4,
        "accuracy": 97.23,
        "description": "European birds (Collins Bird Guide) - previous default",
    },
    "birder-project/naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common": {
        "name": "NaFlex ViT SO150M (Bio-DINOv2, intermediate) EU-Common",
        "species_count": 707,
        "input_size": 336,
        "params_m": 134.3,
        "accuracy": 98.46,
        "naflex": True,
        "description": "European birds (Collins Bird Guide) - most accurate, ~5x slower on CPU",
    },
    "dennisjooo/Birds-Classifier-EfficientNetB2": {
        "name": "EfficientNetB2",
        "species_count": 525,
        "input_size": 224,
        "description": "Global birds (Kaggle dataset)",
    },
}


def is_naflex_model(model_name: str) -> bool:
    """Whether the model expects NaFlex (native aspect ratio) preprocessing at inference."""
    return bool(BIRDER_MODELS.get(model_name, {}).get("naflex")) or "naflex_" in model_name


class BirdClassifier:
    """
    Bird species classifier supporting multiple backends.

    Supports:
    - birder-project models (European birds)
    - HuggingFace transformers models (global birds)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, translation_file: str | None = None, device: str | None = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_birder_model = model_name.startswith("birder-project/")
        # Guards the model state: classify() holds it during inference and
        # reload_model() holds it only to swap in a fully loaded model.
        self._lock = threading.Lock()

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
        """Load a birder-project model, optionally with fine-tuned weights."""
        try:
            import birder
            from birder.inference.classification import infer_image

            birder_model_id = model_name.replace("birder-project/", "")

            self.model, self.model_info = birder.load_pretrained_model(birder_model_id, inference=True)

            # Load fine-tuned weights if available
            if Path(FINETUNED_WEIGHTS).exists():
                self._load_finetuned_weights()

            self.model.to(self.device)

            size = birder.get_size_from_signature(self.model_info.signature)
            if is_naflex_model(model_name):
                # NaFlex models keep the native aspect ratio; token budget matches the reference size
                patch_size = self.model.stem_stride
                max_seq_len = (size[0] // patch_size) * (size[1] // patch_size)
                self.transform = birder.naflex_transform(patch_size, max_seq_len, self.model_info.rgb_stats)
            else:
                self.transform = birder.classification_transform(size, self.model_info.rgb_stats)

            self.id2label = dict(enumerate(self.model_info.class_to_idx.keys()))
            self.infer_func = infer_image

            self.processor = None

        except ImportError as err:
            raise ImportError("birder library not installed. Run: pip install birder") from err

    def _load_finetuned_weights(self) -> None:
        """Load fine-tuned weights over the base model.

        Weights are only applied when they were trained on the currently configured
        base model; after a model switch they are ignored until training runs again.
        """
        try:
            meta = {}
            if Path(FINETUNED_META).exists():
                with open(FINETUNED_META) as f:
                    meta = json.load(f)
            base_model = meta.get("base_model")
            # Weights saved before base_model was recorded were trained on the legacy default
            if (base_model or LEGACY_MODEL) != self.model_name:
                logger.warning(
                    f"Ignoring fine-tuned weights: trained on '{base_model or LEGACY_MODEL}', "
                    f"current model is '{self.model_name}'. Retrain to fine-tune the new model."
                )
                return

            state_dict = torch.load(FINETUNED_WEIGHTS, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            if meta:
                logger.info(
                    f"Loaded fine-tuned weights (trained on {meta.get('num_samples', '?')} "
                    f"samples, val_loss={meta.get('best_val_loss', '?')}, "
                    f"saved {meta.get('saved_at', '?')})"
                )
            else:
                logger.info("Loaded fine-tuned weights")
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned weights, using base model: {e}")

    def reload_model(self) -> bool:
        """Reload the model, picking up fine-tuned weights if available.

        The new model is loaded into a separate object first, so classify()
        keeps using the old model until the swap and never sees a half-loaded one.
        """
        try:
            fresh = BirdClassifier.__new__(BirdClassifier)
            fresh.model_name = self.model_name
            fresh.device = self.device
            fresh.is_birder_model = self.is_birder_model
            if fresh.is_birder_model:
                fresh._load_birder_model(fresh.model_name)
            else:
                fresh._load_transformers_model(fresh.model_name)

            with self._lock:
                for attr in ('model', 'model_info', 'transform', 'id2label', 'infer_func', 'processor'):
                    if hasattr(fresh, attr):
                        setattr(self, attr, getattr(fresh, attr))
            logger.info("Model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False

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
            with self._lock:
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
            top_predictions.append(
                {'species_en': name, 'species_nl': self._translate(name), 'confidence': int(prob * 100)}
            )

        result = {
            'species_en': species_en_clean,
            'species_nl': species_nl,
            'species': f"{species_nl} ({confidence}%)",
            'confidence': confidence,
            'top_predictions': top_predictions,
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
            top_predictions.append(
                {'species_en': name, 'species_nl': self._translate(name), 'confidence': int(prob.item() * 100)}
            )

        result = {
            'species_en': species_en_clean,
            'species_nl': species_nl,
            'species': f"{species_nl} ({confidence}%)",
            'confidence': confidence,
            'top_predictions': top_predictions,
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

    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
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
