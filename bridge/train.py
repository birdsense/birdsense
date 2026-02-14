#!/usr/bin/env python3
"""
BirdSense - Model Fine-Tuning Pipeline

Fine-tunes the ConvNeXt V2 bird classification model using confirmed and
corrected entries from the classification log.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logger = logging.getLogger(__name__)

FINETUNED_DIR = "/data/models"
FINETUNED_WEIGHTS = os.path.join(FINETUNED_DIR, "finetuned_latest.pt")
FINETUNED_META = os.path.join(FINETUNED_DIR, "finetuned_meta.json")


@dataclass
class TrainingStatus:
    """Shared mutable training status, read by API endpoints."""

    running: bool = False
    epoch: int = 0
    total_epochs: int = 0
    samples: int = 0
    loss: float = 0.0
    best_loss: float = float("inf")
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    message: str = ""
    history: list = field(default_factory=list)

    def to_dict(self):
        elapsed = (self.finished_at or time.time()) - self.started_at if self.started_at else 0
        return {
            "running": self.running,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "samples": self.samples,
            "loss": round(self.loss, 5),
            "best_loss": round(self.best_loss, 5) if self.best_loss < float("inf") else None,
            "elapsed_seconds": int(elapsed),
            "error": self.error,
            "message": self.message,
            "history": self.history,
        }


# Global singleton — read by bridge.py API endpoints
training_status = TrainingStatus()


class BirdSenseDataset(Dataset):
    """
    PyTorch dataset built from the classification_log table.

    Uses confirmed + corrected entries only:
    - confirmed: label = species_en (original prediction was correct)
    - corrected: label = species_corrected (user fixed it)
    """

    def __init__(self, entries, class_to_idx):
        self.class_to_idx = class_to_idx
        self.samples = []  # list of (image_path, class_index)

        skipped = 0
        for entry in entries:
            image_path = entry["image_path"]
            if not image_path or not Path(image_path).exists():
                skipped += 1
                continue

            # Determine the correct label
            if entry["status"] == "corrected" and entry.get("species_corrected"):
                species = entry["species_corrected"]
            else:
                species = entry["species_en"]

            species_key = self._find_class_key(species)
            if species_key is None:
                logger.debug(f"Skipping entry: species '{species}' not in model vocabulary")
                skipped += 1
                continue

            idx = class_to_idx[species_key]
            self.samples.append((image_path, idx))

        logger.info(
            f"Dataset: {len(self.samples)} usable samples, {skipped} skipped "
            f"(missing file or unknown species)"
        )

    def _find_class_key(self, species_name):
        """Match a species name to the model's class_to_idx keys."""
        if species_name in self.class_to_idx:
            return species_name

        normalized = species_name.lower().replace(" ", "_")
        if normalized in self.class_to_idx:
            return normalized

        title = species_name.replace("_", " ").title()
        if title in self.class_to_idx:
            return title

        for key in self.class_to_idx:
            if key.lower().replace("_", " ") == species_name.lower().replace("_", " "):
                return key

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return image, label


class TransformSubset(Dataset):
    """Wraps a Subset to apply a specific transform per split."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _extract_rgb_stats(rgb_stats):
    """Extract mean and std from birder's rgb_stats (handles dict, tuple, or named tuple)."""
    if isinstance(rgb_stats, dict):
        mean = rgb_stats["mean"]
        std = rgb_stats["std"]
    elif hasattr(rgb_stats, "mean") and hasattr(rgb_stats, "std"):
        mean = rgb_stats.mean
        std = rgb_stats.std
    else:
        mean, std = rgb_stats

    return [float(v) for v in mean], [float(v) for v in std]


def get_training_transform(size, rgb_stats):
    """Create a training transform with data augmentation."""
    mean, std = _extract_rgb_stats(rgb_stats)
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _unfreeze_head(model):
    """Find and unfreeze the classification head of the model."""
    head_names = ["classifier", "head", "fc", "linear"]
    head_params = []

    for name in head_names:
        head_module = getattr(model, name, None)
        if head_module is not None and hasattr(head_module, "parameters"):
            for param in head_module.parameters():
                param.requires_grad = True
                head_params.append(param)
            logger.info(
                f"Unfroze head module: '{name}' "
                f"({sum(p.numel() for p in head_params)} parameters)"
            )
            return head_params

    # Fallback: unfreeze last 2 parameter groups
    all_params = list(model.parameters())
    if len(all_params) >= 2:
        for param in all_params[-2:]:
            param.requires_grad = True
            head_params.append(param)
        logger.info(
            f"Unfroze last 2 parameter groups as fallback "
            f"({sum(p.numel() for p in head_params)} parameters)"
        )

    return head_params


def _split_parameters(model):
    """Split model parameters into head and backbone groups."""
    head_names = ["classifier", "head", "fc", "linear"]
    head_params = []
    backbone_params = []

    head_param_ids = set()
    for name in head_names:
        head_module = getattr(model, name, None)
        if head_module is not None and hasattr(head_module, "parameters"):
            for param in head_module.parameters():
                head_params.append(param)
                head_param_ids.add(id(param))
            break

    for param in model.parameters():
        if id(param) not in head_param_ids:
            backbone_params.append(param)

    if not head_params:
        return list(model.parameters()), []

    return head_params, backbone_params


def _save_finetuned(model, base_model_name, epochs, freeze_backbone,
                    num_samples, best_loss):
    """Save fine-tuned model weights and metadata."""
    os.makedirs(FINETUNED_DIR, exist_ok=True)

    torch.save(model.state_dict(), FINETUNED_WEIGHTS)

    meta = {
        "base_model": base_model_name,
        "epochs": epochs,
        "freeze_backbone": freeze_backbone,
        "num_samples": num_samples,
        "best_val_loss": round(best_loss, 5),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    with open(FINETUNED_META, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Fine-tuned model saved to {FINETUNED_WEIGHTS}")


def run_training(
    model_name: str = "birder-project/convnext_v2_tiny_eu-common",
    epochs: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    freeze_backbone: bool = True,
    val_split: float = 0.2,
    species: str = "",
):
    """Main training function. Runs in a background thread."""
    global training_status
    species_label = f" for '{species}'" if species else ""
    training_status = TrainingStatus(
        running=True,
        total_epochs=epochs,
        started_at=time.time(),
        message=f"Loading training data{species_label}...",
    )

    try:
        _run_training_inner(model_name, epochs, learning_rate, batch_size,
                            freeze_backbone, val_split, species)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        training_status.error = str(e)
        training_status.message = f"Training failed: {e}"
    finally:
        training_status.running = False
        training_status.finished_at = time.time()


def _run_training_inner(model_name, epochs, lr, batch_size, freeze_backbone, val_split, species):
    global training_status
    from database import ClassificationLog

    # Step 1: Gather training data
    confirmed = ClassificationLog.get_entries(limit=100000, status="confirmed")
    corrected = ClassificationLog.get_entries(limit=100000, status="corrected")
    all_entries = confirmed + corrected

    # Filter to a single species if specified
    if species:
        species_lower = species.lower()
        all_entries = [
            e for e in all_entries
            if (e.get("species_corrected") or e.get("species_en", "")).lower() == species_lower
        ]
        logger.info(f"Filtered to species '{species}': {len(all_entries)} entries")

    min_samples = 2 if species else 5
    if len(all_entries) < min_samples:
        raise ValueError(
            f"Not enough training data: {len(all_entries)} entries "
            f"(need at least {min_samples} confirmed/corrected classifications"
            f"{f' for {species}' if species else ''})"
        )

    training_status.samples = len(all_entries)
    training_status.message = f"Found {len(all_entries)} training samples. Loading model..."

    # Step 2: Load the base model for training
    import birder

    birder_model_id = model_name.replace("birder-project/", "")
    model, model_info = birder.load_pretrained_model(birder_model_id, inference=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class_to_idx = model_info.class_to_idx
    size = birder.get_size_from_signature(model_info.signature)

    # Step 3: Create dataset
    train_transform = get_training_transform(size, model_info.rgb_stats)
    val_transform = birder.classification_transform(size, model_info.rgb_stats)

    full_dataset = BirdSenseDataset(all_entries, class_to_idx)

    if len(full_dataset) < 5:
        raise ValueError(
            f"Only {len(full_dataset)} usable samples after filtering "
            f"(images must exist and species must be in the model's vocabulary)"
        )

    # Split into train/val
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        TransformSubset(train_dataset, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        TransformSubset(val_dataset, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    mode_label = "head-only" if freeze_backbone else "full"
    training_status.message = (
        f"Training: {train_size} train / {val_size} val samples, "
        f"{mode_label} fine-tuning on {device}"
    )

    # Step 4: Configure optimizer
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        head_params = _unfreeze_head(model)
        if not head_params:
            raise ValueError(
                "Could not find classifier head to unfreeze. "
                "Try freeze_backbone=false for full fine-tuning."
            )
        optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=1e-2)
    else:
        head_params, backbone_params = _split_parameters(model)
        param_groups = [
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Step 5: Training loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        training_status.epoch = epoch + 1
        training_status.message = f"Epoch {epoch + 1}/{epochs} - training..."

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        training_status.loss = avg_val_loss
        training_status.history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(avg_val_loss, 5),
            "val_acc": round(val_acc, 4),
        })

        logger.info(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.2%}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            training_status.best_loss = best_val_loss
            _save_finetuned(model, model_name, epochs, freeze_backbone,
                            len(all_entries), best_val_loss)
            training_status.message = (
                f"Epoch {epoch + 1}/{epochs} - new best! val_loss={avg_val_loss:.4f}"
            )

    training_status.message = (
        f"Training complete! Best val_loss={best_val_loss:.4f}. "
        f"Reloading model..."
    )
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")

    # Auto-reload the model after training
    try:
        from classifier import BirdClassifier
        BirdClassifier.reload_model()
        training_status.message = (
            f"Training complete! Best val_loss={best_val_loss:.4f}. "
            f"Model reloaded successfully."
        )
        logger.info("Model auto-reloaded after training")
    except Exception as e:
        logger.warning(f"Auto-reload failed: {e}")
        training_status.message = (
            f"Training complete! Best val_loss={best_val_loss:.4f}. "
            f"Auto-reload failed, click Reload Model manually."
        )
