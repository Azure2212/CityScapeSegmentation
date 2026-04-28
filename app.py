"""Unified Flask entrypoint for live segmentation and urban-scene analysis."""

import os
# Some local macOS OpenMP builds raise duplicate-library errors when the app
# loads Torch and other native dependencies in the same process.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
# The server renders figures only as files, so a non-interactive backend keeps
# startup compatible with headless execution.
matplotlib.use("Agg")

import io
import sys
import types
import base64
import numpy as np
import cv2
import torch
import gdown
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from application import CLASSES, CONFIG_CMAP, DEVICE, NUM_CLASSES
from models import load_UNet, load_FCN, load_DeepLabV3, load_LightSeg, load_SwinV2B, load_YOLOv11Seg
from utils.urban_scene_analysis import analyze_urban_scene, compare_scene_analyses, count_connected_components
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# The root page starts with a practical subset of classes enabled for the
# overlay so the first run highlights the dominant street-scene categories.
DEFAULT_SELECTED_CLASSES = (0, 1, 2, 8, 10, 11, 13)
# The root page starts with one primary model selected so the unified checkbox
# list can drive either a single-model run or a multi-model comparison.
DEFAULT_SELECTED_MODELS = ("UNet",)

# The model registry is the single source of truth for labels shown in the UI,
# checkpoint download URLs, loader functions, and input resolution.
MODEL_REGISTRY = {
    "UNet": {
        "label": "UNet",
        # Empty URLs disable a model without removing it from the registry.
        "url":"https://drive.google.com/uc?export=download&id=1Lc0hI4WxPoT56syH-SMGUBYU8mSird38",
        "loader": lambda buf: _load_unet(buf, use_cbam=False),
        "input_size": 224,
    },
    "UNet_CBAM": {
        "label": "UNet + CBAM",
        "url": "https://drive.google.com/uc?export=download&id=1NGCwzJ1UR_vmrQQe69j_Lb3XvlKFOtj8",
        "loader": lambda buf: _load_unet(buf, use_cbam=True),
        "input_size": 224,
    },
    "FCN": {
        "label": "FCN (ResNet-50)",
        "url": "https://drive.google.com/uc?export=download&id=1GtnAoCyQJZUHgfaH9ZAS99YNFd-Roa8f",
        "loader": lambda buf: _load_generic(buf, load_FCN, num_classes=NUM_CLASSES),
        "input_size": 224,
    },
    "DeepLabV3": {
        "label": "DeepLabV3",
        "url": "https://drive.google.com/uc?export=download&id=1PG6XjfOG4LZn9kARZaBptMcVnojcgB99",
        "loader": lambda buf: _load_generic(buf, load_DeepLabV3, num_classes=NUM_CLASSES),
        "input_size": 224,
    },
    "LightSeg": {
        "label": "LightSeg",
        "url": "https://drive.google.com/uc?export=download&id=1fMm-yde5QyLAz5ph9FbY9K8gkgI9aii-",
        "loader": lambda buf: _load_generic(buf, load_LightSeg, num_classes=NUM_CLASSES),
        "input_size": 224,
    },
    "SwinV2B": {
        "label": "SwinV2B",
        "url": "https://drive.google.com/uc?export=download&id=15ZJbxDn7xph7zyXJOBdu417uVv4BIU1T",
        "loader": lambda buf: _load_generic(buf, load_SwinV2B, num_classes=NUM_CLASSES),
        "input_size": 256,
    },
}

# The in-process cache avoids re-downloading and re-instantiating checkpoints on
# every request once a model has been used or preloaded.
_model_cache = {}

def get_transform(input_size: int):
    """Build the inference transform for one model input resolution."""
    return A.Compose([A.Resize(input_size, input_size), ToTensorV2()])


def _load_unet(buf, use_cbam=False):
    """Restore a U-Net checkpoint from an in-memory buffer."""
    state = torch.load(buf, map_location=str(DEVICE), weights_only=False)
    model = load_UNet(n_channels=3, cls_classes=NUM_CLASSES, use_cbam=use_cbam)
    model.load_state_dict(state["net"])
    return model


def _load_generic(buf, loader_fn, num_classes):
    """Restore a non-U-Net checkpoint from an in-memory buffer."""
    state = torch.load(buf, map_location=str(DEVICE), weights_only=False)
    model = loader_fn(num_classes=num_classes)
    model.load_state_dict(state["net"])
    return model


def get_model(key: str):
    """Return a ready-to-run model, downloading weights on first use if needed."""
    if key in _model_cache:
        return _model_cache[key]

    entry = MODEL_REGISTRY.get(key)
    if entry is None:
        raise ValueError(f"Unknown model: {key}")
    if not entry["url"]:
        raise ValueError(f"No pretrained URL configured for model: {key}")

    print(f"Downloading weights for {key} ...")
    buf = io.BytesIO()
    gdown.download(entry["url"], buf, quiet=False)
    buf.seek(0)

    model = entry["loader"](buf)

    # Some Swin checkpoints expose the final feature tensor with channel-last
    # layout, so the helper is normalized here before inference requests use it.
    if hasattr(model, "_to_bchw"):
        def _fixed_to_bchw(_, t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 4 and t.shape[-1] > t.shape[1]:
                t = t.permute(0, 3, 1, 2).contiguous()
            return t
        model._to_bchw = types.MethodType(_fixed_to_bchw, model)

    model.to(DEVICE)
    model.eval()
    _model_cache[key] = model
    print(f"{key} ready.")
    return model


def preload_available_models():
    """Eagerly warm the model cache for every registry entry with a checkpoint URL."""
    print("Pre-loading all available models ...")
    for _key, _entry in MODEL_REGISTRY.items():
        if _entry["url"]:
            get_model(_key)
    print("All models ready.")


# Tests disable preload so the suite does not trigger network downloads during import.
if os.environ.get("CITYSCAPES_PRELOAD_MODELS", "1") != "0":
    preload_available_models()
else:
    print("Skipping model preload because CITYSCAPES_PRELOAD_MODELS=0.")


def _predict_full_mask(image_bgr: np.ndarray, model_key: str) -> np.ndarray:
    """Run one model and resize its predicted mask back to the source image size."""
    # Resolve the requested model from the registry/cache before any image work
    # so invalid model keys fail before preprocessing begins.
    model = get_model(model_key)
    # Each architecture is evaluated at the input size it was trained for.
    input_size = MODEL_REGISTRY[model_key]["input_size"]
    transform = get_transform(input_size)

    # The model expects normalized RGB tensors at the architecture-specific
    # input resolution declared in the registry.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Albumentations returns a channel-first tensor; the batch dimension is
    # added here because the model always expects batched input.
    tensor = transform(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        # Argmax converts per-class logits into the single winning class ID for
        # each pixel in the resized model-output grid.
        pred_mask = torch.argmax(model(tensor), dim=1)

    # The raw prediction is generated at model resolution, so it is expanded to
    # the original image geometry before overlay rendering and scene analysis.
    mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    orig_h, orig_w = image_bgr.shape[:2]
    # Nearest-neighbor resizing preserves discrete class IDs during expansion.
    return cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def _encode_segmentation_png(
    image_bgr: np.ndarray,
    mask_full: np.ndarray,
    selected_classes: set[int],
) -> str:
    """Render a PNG overlay that colors only the selected semantic classes."""
    # The configured Matplotlib colormap is converted once into 8-bit RGB
    # colors so the predicted class IDs can be mapped directly onto pixels.
    cmap_colors = (np.array(CONFIG_CMAP.colors) * 255).astype(np.uint8)
    # The original image is kept as the visual base layer and only selected
    # semantic regions are recolored on top of it.
    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = orig_rgb.copy()

    # Unselected classes remain unchanged so the overlay can focus attention on
    # the semantic categories currently toggled in the sidebar.
    for class_id in selected_classes:
        if 0 <= class_id < len(cmap_colors):
            # Every pixel predicted as the selected class is replaced by the
            # configured semantic color while all other pixels keep the source image.
            result[mask_full == class_id] = cmap_colors[class_id]

    # The frontend expects a base64-encoded PNG string that can be embedded
    # directly into the rendered `<img>` element.
    _, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _selected_overlay_classes(selected_class_ids: list[int]) -> set[int]:
    """Resolve the overlay class selection sent by the frontend."""
    if not selected_class_ids:
        # A missing field means "use all classes" to preserve the original API
        # contract for clients that do not send explicit overlay selections.
        return set(CLASSES.keys())
    # Invalid IDs are discarded so the overlay renderer never receives class
    # values that are outside the known Cityscapes label set.
    return {class_id for class_id in selected_class_ids if class_id in CLASSES}


def _comparison_models(base_model: str, requested_models: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Resolve requested comparison models and record any skipped entries."""
    selected = []
    skipped = []
    seen = set()

    # The base model is always included first so comparison output can treat it
    # as the reference analysis alongside any additional requested models.
    for model_key in [base_model, *requested_models]:
        # Duplicate requests are ignored so each model is analyzed at most once.
        if model_key in seen:
            continue
        seen.add(model_key)

        entry = MODEL_REGISTRY.get(model_key)
        if entry is None:
            # Unknown keys are reported back to the client instead of raising so
            # the main analysis can still complete.
            skipped.append({"key": model_key, "label": model_key, "reason": "unknown model"})
            continue
        if not entry["url"]:
            # Models without an active checkpoint URL stay visible in the UI but
            # are skipped during runtime comparison.
            skipped.append({"key": model_key, "label": entry["label"], "reason": "model unavailable"})
            continue
        selected.append(model_key)

    return selected, skipped


def _requested_model_selection() -> tuple[str, list[str]]:
    """Resolve the primary model and any comparison models from the request."""
    selected_models = list(dict.fromkeys(request.form.getlist("models[]")))
    if selected_models:
        return selected_models[0], selected_models[1:]

    model_key = request.form.get("model", DEFAULT_SELECTED_MODELS[0])
    compare_models = list(dict.fromkeys(request.form.getlist("compare_models[]")))
    return model_key, compare_models


def _build_reasoning(
    mask_full: np.ndarray,
    selected_classes: set[int],
    model_key: str,
    summary: str,
    min_area: int = 25,
) -> str:
    """Combine the planning summary with the legacy per-class reasoning block."""
    total_pixels = mask_full.size
    model_label = MODEL_REGISTRY[model_key]["label"]

    # The merged reasoning string keeps the original text panel useful by
    # leading with the new planning summary and then appending the older
    # per-class count/coverage lines.
    lines = [summary, "", f"The input image was analysed with {model_label}:"]
    if not selected_classes:
        lines.append("+ Overlay disabled for this run.")

    for class_id in sorted(selected_classes):
        # Reasoning lines still respect the overlay selection because the panel
        # explains what the user chose to visualize, not the hidden full-class list.
        class_pixels = int(np.sum(mask_full == class_id))
        if class_pixels == 0:
            continue

        # Connected components provide an approximate object count for countable
        # classes and a coarse region count for larger "stuff" selections.
        count = count_connected_components(mask_full, class_id=class_id, min_area=min_area)
        pct = class_pixels / total_pixels * 100
        lines.append(f"+ {count} {CLASSES[class_id]}(s) found! ({pct:.1f}%)")

    return "\n".join(lines)


@app.route("/")
def index():
    """Render the unified segmentation and planning dashboard."""
    # The template receives the same registry metadata used by the backend so
    # the available-model UI stays synchronized with the API.
    models = [{"key": k, "label": v["label"], "available": bool(v["url"])}
              for k, v in MODEL_REGISTRY.items()]
    return render_template(
        "index.html",
        models=models,
        classes=CLASSES,
        default_checked_classes=DEFAULT_SELECTED_CLASSES,
        default_selected_models=DEFAULT_SELECTED_MODELS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Return segmentation output, planning metrics, and optional model comparison."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    # The selected model drives inference, the class list drives only overlay
    # coloring, and any additional selected models request extra analyses on
    # the same image. The checkbox-based UI sends models[] while older clients
    # can still send model + compare_models[].
    model_key, compare_models = _requested_model_selection()
    selected_classes = request.form.getlist("classes[]", type=int)

    # Decode the uploaded image into OpenCV BGR format so the same image buffer
    # can be reused for overlay rendering and model preprocessing.
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Cannot decode image"}), 400

    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    try:
        # This full-resolution mask is the single source of truth for both the
        # scene-analysis pipeline and any overlay derived from the prediction.
        mask_full = _predict_full_mask(image, model_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Planning metrics are always computed from the full predicted mask, while
    # the sidebar selection affects only overlay visualization.
    min_area = 25
    # The analysis layer reads the complete mask so planning scores do not
    # change when the user hides classes in the overlay.
    analysis = analyze_urban_scene(mask_full, classes=CLASSES, min_area=min_area)
    # Overlay selection is resolved after analysis because it is a visualization
    # concern rather than an analysis concern.
    selected_set = _selected_overlay_classes(selected_classes)

    # The response keeps the original overlay/reasoning keys and extends them
    # with the structured scene-analysis payload used by the dashboard panels.
    payload = {
        "model": model_key,
        "model_label": MODEL_REGISTRY[model_key]["label"],
        "segmentation_image": _encode_segmentation_png(image, mask_full, selected_set),
        "reasoning": _build_reasoning(mask_full, selected_set, model_key, analysis["summary"], min_area=min_area),
        **analysis,
    }

    if compare_models:
        # Comparison requests are normalized before any extra inference runs so
        # duplicates and unavailable entries are handled consistently.
        comparison_keys, skipped_models = _comparison_models(model_key, compare_models)
        # The already computed primary analysis becomes the comparison baseline.
        compared_analyses = {model_key: analysis}

        # Each additional comparison model runs through the same full-mask
        # analysis pipeline so cross-model differences are measured consistently.
        for compare_key in comparison_keys:
            if compare_key == model_key:
                continue

            try:
                # The same decoded image is reused so differences reflect the
                # segmentation model rather than any preprocessing variation.
                compare_mask = _predict_full_mask(image, compare_key)
            except Exception as exc:  # pragma: no cover - patched in endpoint tests
                skipped_models.append(
                    {
                        "key": compare_key,
                        "label": MODEL_REGISTRY[compare_key]["label"],
                        "reason": str(exc),
                    }
                )
                continue

            # Each comparison model is reduced to the same structured analysis
            # schema used for the primary prediction.
            compared_analyses[compare_key] = analyze_urban_scene(compare_mask, classes=CLASSES, min_area=min_area)

        # The comparison payload summarizes shared tags, per-class spreads, and
        # disagreement notes across the successfully analyzed model set.
        payload["model_comparison"] = compare_scene_analyses(
            compared_analyses,
            model_labels={key: MODEL_REGISTRY[key]["label"] for key in compared_analyses},
            skipped_models=skipped_models,
        )

    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10796))
    app.run(host="0.0.0.0", port=port)
