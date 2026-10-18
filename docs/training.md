# Model Training

Fine-tune the configured bird classification model (default: Birder ViT Parallel EU-Common) using confirmed and corrected data from the classification log.

## Training Data

Every time you **confirm** or **correct** a classification in the Log, you create a labeled training example:

| Log Status | Label Used for Training |
|---|---|
| Confirmed | `species_en` (the AI was correct) |
| Corrected | `species_corrected` (your correction) |
| Pending | **Not used** (unverified) |

Each example is a pair of: **image** (`image_path`) + **correct species** (label).

You need at least 5 confirmed/corrected entries to start training. More data (especially corrections) leads to better results.

## How It Works Step by Step

### 1. Data Collection

All confirmed + corrected entries are pulled from the `classification_log` database table. Each species name is matched to the model's vocabulary of 707 species. Entries where the image file no longer exists or the species is not in the model's vocabulary are skipped.

### 2. Dataset Split

80% of the data goes to training, 20% to validation. The validation set is never trained on — it exists only to measure whether the model is actually improving (and not just memorizing the training images).

### 3. Model Loading

The pretrained base model selected by `MODEL_NAME` (707 European bird species, trained on ~570k images) is loaded. This model can already recognize birds — we're just making it *better* for your specific situation.

### 4. Freeze Strategy

- **Head-only** (default): All layers are frozen except the final classifier layer (~272K parameters for ViT Parallel, ~544K for ConvNeXt V2 / Hiera). The model keeps its "knowledge of bird shapes" but re-learns the decision boundaries. Fast, works well with little data.
- **Full**: All layers are trained, but the backbone gets a 10x lower learning rate than the head. The model can learn more subtle patterns, but needs more data and takes longer.

### 5. Training Loop

Per epoch:

```
For each batch of images:
  1. Apply augmentation (random crop, flip, color shift, rotation)
     → prevents the model from only learning the exact same images
  2. Forward pass: model predicts species
  3. Compute loss (CrossEntropyLoss): how far off is the prediction?
  4. Backward pass: compute gradients
  5. Optimizer step (AdamW): adjust weights
```

### 6. Validation

After each epoch, the model is tested on the 20% validation data (without augmentation, without gradients). This produces the `val_loss` and `val_acc`.

### 7. Best Model Saving

If the `val_loss` of this epoch is lower than all previous ones, the weights are saved to `/data/models/finetuned_latest.pt`. This way you always keep the best model, not the last one.

## Base Models

Any `birder-project/*eu-common` model can be fine-tuned. They all share the same 707-species label set (same order), so logged labels and Dutch names keep working when you switch. Accuracy is top-1 on the eu-common validation set ([Birder leaderboard](https://huggingface.co/spaces/birder-project/leaderboard)); CPU latency is a rough single-image figure measured on a desktop CPU with 4 threads.

| `MODEL_NAME` | Params | Accuracy | CPU latency | Download |
|---|---|---|---|---|
| `birder-project/vit_parallel_s16_18x2_ls_avg_data2vec-intermediate-eu-common` (default) | 65M | 98.0% | ~0.7 s | 259 MB |
| `birder-project/hiera_abswin_base_mim-intermediate-eu-common` | 51M | 97.8% | ~0.5 s | 204 MB |
| `birder-project/convnext_v2_tiny_intermediate-eu-common` | 28M | 97.4% | ~0.3 s | 114 MB |
| `birder-project/convnext_v2_tiny_eu-common` (previous default) | 28M | 97.2% | ~0.3 s | 114 MB |
| `birder-project/naflex_vit_reg4_so150m_p14_ls_dino-v2-bio-intermediate-eu-common` | 134M | 98.5% | ~1.5 s+ | 538 MB |

Pretrained weights are downloaded on first start into `/data/models/birder` (the `birdsense-models` volume).

Fine-tuned weights belong to one base model: `finetuned_meta.json` records `base_model`, and the classifier ignores the weights (with a warning in the log) when `MODEL_NAME` points to a different model. After switching models, start a training run (or confirm/correct a log entry, which triggers auto-training) to fine-tune the new model. To go back to the previous behaviour, set `MODEL_NAME=birder-project/convnext_v2_tiny_eu-common`.

The NaFlex model uses native-aspect-ratio preprocessing at inference time; fine-tuning uses fixed 336x336 crops, which the model also supports.

## After Training

When training is started from the API/UI or by auto-training, the running classifier is reloaded automatically once training finishes. If that reload fails, the weights are saved but **not yet active** until you click **Reload Model**:

1. `classifier.reload_model()` is called
2. The base model is reloaded
3. The fine-tuned weights are loaded on top (`load_state_dict`)
4. All subsequent classifications use the improved model

On a **container restart**, fine-tuned weights are automatically loaded (if they exist), because `_load_birder_model()` checks for the file at startup.

## Understanding the Results Table

| Column | Meaning |
|---|---|
| **Train Loss** | How far off the model is on training data (lower = better) |
| **Train Acc** | % correct on training data |
| **Val Loss** | How far off the model is on data it hasn't seen |
| **Val Acc** | % correct on unseen data — this is the real metric |

If `val_loss` rises while `train_loss` falls, the model is **overfitting** (memorizing). Stop earlier or use head-only mode.

## Default Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| Epochs | 10 | Number of full passes over the dataset |
| Learning rate | 0.0001 | Step size for weight updates |
| Batch size | 8 | Images per training step (lower = less memory) |
| Freeze backbone | true | Head-only (true) vs full fine-tuning (false) |
| Validation split | 0.2 | 20% of data reserved for validation |
| Optimizer | AdamW | With weight decay 1e-2 |
| Scheduler | CosineAnnealing | Smooth learning rate decay |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/train` | POST | Start training. Body: `{epochs, learning_rate, batch_size, freeze_backbone}` |
| `/api/train/status` | GET | Get current training status (epoch, loss, progress) |
| `/api/train/reload` | POST | Reload classifier with fine-tuned weights |

## File Locations

| File | Purpose |
|---|---|
| `bridge/train.py` | Training pipeline (dataset, training loop, status tracking) |
| `bridge/classifier.py` | Model loading with fine-tuned weight support |
| `/data/models/finetuned_latest.pt` | Saved fine-tuned model weights |
| `/data/models/finetuned_meta.json` | Training metadata (samples, loss, date) |
