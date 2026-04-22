# CityScapeSegmentation

CityScapeSegmentation is a Flask-based semantic-segmentation project for urban
street scenes. The application loads trained segmentation models, predicts a
pixel-level Cityscapes mask for an uploaded image, renders a selectable overlay,
and derives an explainable urban-scene analysis layer from the predicted mask.

The current app is unified around a single web interface:

- `GET /` renders the main application.
- `POST /predict` runs segmentation, overlay rendering, heuristic scene
  analysis, and optional cross-model comparison.

## Main Capabilities

- Semantic segmentation for Cityscapes-style street-scene classes.
- Overlay rendering with per-class visibility controls.
- Heuristic scene-analysis outputs derived from the full predicted mask:
  planning scores, layout profile, object counts, scene tags, warnings, and
  summary text.
- Optional model comparison that contrasts multiple segmentation models on the
  same uploaded image.
- Training and evaluation utilities for the supported segmentation models.

## Repository Structure

- `app.py`: unified Flask entrypoint and live inference API.
- `application.py`: shared runtime configuration, device selection, class map,
  and color map definitions used by the app.
- `templates/index.html`: main interface for upload, visualization, reasoning,
  and appended scene-analysis panels.
- `utils/urban_scene_analysis.py`: explainable post-processing pipeline that
  converts a segmentation mask into higher-level scene descriptors.
- `models/`: model definitions and weight-loading helpers.
- `trainer/`, `train.py`: training pipeline entrypoint and trainer logic.
- `utils/datasets/`, `utils/losses.py`, `utils/metrics.py`,
  `utils/trainingStrategies/`: dataset loading and training utilities.
- `evaluations/`: evaluation plots and saved qualitative prediction helpers.
- `tests/`: unit tests for the unified endpoint and the scene-analysis logic.
- `image2Test/`: sample images for quick local inference checks.

## Prerequisites

- Python 3.10+ is recommended.
- A virtual environment is recommended for dependency isolation.
- PyTorch-compatible hardware is optional; the app falls back to CPU when CUDA
  is unavailable.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data And Checkpoints

Dataset:

- Cityscapes segmentation dataset mirror:
  [Kaggle](https://www.kaggle.com/datasets/azurebob2/cityscapesegmentation)

Pretrained checkpoints:

- Project checkpoint folder:
  [Google Drive](https://drive.google.com/drive/folders/17qMjLDDrNTJZU9snCqFihdSPesdOLjPQ)

The Flask app downloads configured pretrained weights on first use through the
URLs defined in `app.py`. Training expects a local dataset path configured in
`configs.json` or passed to `train.py` as an override.

## Run The Unified App

Start the Flask application from the repository root:

```bash
python app.py
```

Then open:

- `http://127.0.0.1:5000/`

The root route contains the full workflow:

1. Upload an image.
2. Select the segmentation model.
3. Choose which classes should appear in the rendered overlay.
4. Optionally request comparison models.
5. Run inference and inspect the segmentation result, reasoning text, planning
   summary, interpretation panels, and detailed scene metrics on the same page.

## Unified API Surface

`POST /predict` accepts:

- `image`
- `model`
- `classes[]` for overlay selection
- optional `compare_models[]`

The response preserves the original keys:

- `segmentation_image`
- `reasoning`

The response also includes structured scene-analysis fields:

- `summary`
- `planning_scores`
- `spatial_flags`
- `layout_profile`
- `region_stats`
- `relation_flags`
- `scene_tags`
- `analysis_warnings`
- `group_stats`
- `object_counts`
- `class_stats`
- optional `model_comparison`

Important behavior:

- `classes[]` affects only the rendered overlay.
- Planning metrics always run on the full predicted mask, not the filtered
  overlay selection.

## Training Overview

The training entrypoint is `train.py`. At a high level, the training workflow:

1. Loads the JSON config from `configs.json`.
2. Applies command-line overrides such as dataset path, image size, batch size,
   and epoch count.
3. Builds train, validation, and test dataloaders.
4. Instantiates the requested model architecture.
5. Trains through `trainer/unet_trainer.py`, which handles losses, metrics,
   checkpointing, scheduler stepping, and early stopping.
6. Runs post-training evaluation utilities to save plots and sample
   predictions.

Example:

```bash
python train.py --model UNet --cityscape_path /path/to/cityscapes
```

For cluster execution, the repository also includes `train_sbatch`.

## Tests

Run the test suite with the project virtual environment:

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests set `CITYSCAPES_PRELOAD_MODELS=0` so importing `app.py` does not
trigger checkpoint downloads during test startup.

## Scene-Analysis Interpretation Limits

The scene-analysis layer in `utils/urban_scene_analysis.py` is heuristic and
segmentation-driven. It is designed to turn pixel predictions into transparent,
reviewable signals such as layout priors, region counts, and planning-oriented
tags. These outputs are useful for demonstration, inspection, and comparative
analysis, but they are not surveyed urban-planning measurements and should not
be treated as ground-truth scene geometry or formal planning assessment.

## Additional Notes

- Sample images for local testing are stored in `image2Test/`.
- The deployed project reference linked in earlier materials is:
  [Hugging Face Space](https://huggingface.co/spaces/Azure2212/CityScapeSegmentationWebsite)
