import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
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
from application import countObject, CLASSES, CONFIG_CMAP, DEVICE, NUM_CLASSES
from models import load_UNet, load_FCN, load_DeepLabV3, load_LightSeg, load_SwinV2B, load_YOLOv11Seg
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────
# Add Google Drive URLs here when you have them. Leave url="" to disable a model.
MODEL_REGISTRY = {
    "UNet": {
        "label": "UNet",
        #"url": "https://drive.google.com/uc?export=download&id=1nGHMD7RMPZOuqYm-9xAStGqJSyI28Gzd",
        "url":"https://drive.google.com/uc?export=download&id=1Lc0hI4WxPoT56syH-SMGUBYU8mSird38",
        "loader": lambda buf: _load_unet(buf, use_cbam=False),
        "input_size": 224,
    },
    "UNet_CBAM": {
        "label": "UNet + CBAM",
        "url": "https://drive.google.com/uc?export=download&id=1NGCwzJ1UR_vmrQQe69j_Lb3XvlKFOtj8",  # add your Drive URL here
        "loader": lambda buf: _load_unet(buf, use_cbam=True),
        "input_size": 224,
    },
    "FCN": {
        "label": "FCN (ResNet-50)",
        "url": "https://drive.google.com/uc?export=download&id=1GtnAoCyQJZUHgfaH9ZAS99YNFd-Roa8f",  # add your Drive URL here
        "loader": lambda buf: _load_generic(buf, load_FCN, num_classes=NUM_CLASSES),
        "input_size": 224,
    },
    "DeepLabV3": {
        "label": "DeepLabV3",
        "url": "https://drive.google.com/uc?export=download&id=1PG6XjfOG4LZn9kARZaBptMcVnojcgB99",  # add your Drive URL here
        "loader": lambda buf: _load_generic(buf, load_DeepLabV3, num_classes=NUM_CLASSES),
        "input_size": 224,
    },
    "LightSeg": {
        "label": "LightSeg",
        "url": "https://drive.google.com/uc?export=download&id=1fMm-yde5QyLAz5ph9FbY9K8gkgI9aii-",  # add your Drive URL here
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

# Cache: model_key -> loaded model
_model_cache = {}

def get_transform(input_size: int):
    return A.Compose([A.Resize(input_size, input_size), ToTensorV2()])


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_unet(buf, use_cbam=False):
    state = torch.load(buf, map_location=str(DEVICE), weights_only=False)
    model = load_UNet(n_channels=3, cls_classes=NUM_CLASSES, use_cbam=use_cbam)
    model.load_state_dict(state["net"])
    return model


def _load_generic(buf, loader_fn, num_classes):
    state = torch.load(buf, map_location=str(DEVICE), weights_only=False)
    model = loader_fn(num_classes=num_classes)
    model.load_state_dict(state["net"])
    return model


def get_model(key: str):
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

    # Fix inverted NHWC→NCHW condition in SwinV2B._to_bchw
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
    """Warm the model cache unless tests explicitly disable startup downloads."""
    print("Pre-loading all available models ...")
    for _key, _entry in MODEL_REGISTRY.items():
        if _entry["url"]:
            get_model(_key)
    print("All models ready.")


if os.environ.get("CITYSCAPES_PRELOAD_MODELS", "1") != "0":
    preload_available_models()
else:
    print("Skipping model preload because CITYSCAPES_PRELOAD_MODELS=0.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    models = [{"key": k, "label": v["label"], "available": bool(v["url"])}
              for k, v in MODEL_REGISTRY.items()]
    return render_template("index.html", models=models)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    selected_classes = request.form.getlist("classes[]", type=int)
    model_key = request.form.get("model", "UNet")

    # Decode uploaded image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Cannot decode image"}), 400

    # Load requested model
    try:
        m = get_model(model_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Run inference
    input_size = MODEL_REGISTRY[model_key]["input_size"]
    tf = get_transform(input_size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = tf(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    m.eval()
    with torch.no_grad():
        pred_mask = torch.argmax(m(tensor), dim=1)  # (1, H, W)

    mask_np = pred_mask.squeeze(0).cpu().numpy()    # (224, 224)

    # Upscale mask to original image size so no content is lost
    orig_h, orig_w = image.shape[:2]
    mask_full = cv2.resize(mask_np.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Build composite: selected classes → segmentation color, others → original
    cmap_colors = (np.array(CONFIG_CMAP.colors) * 255).astype(np.uint8)  # (20, 3)
    orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = orig_rgb.copy()
    selected_set = set(selected_classes) if selected_classes else set(CLASSES.keys())
    for class_id in selected_set:
        region = mask_full == class_id
        if class_id < len(cmap_colors):
            result[region] = cmap_colors[class_id]

    _, buf = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    seg_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    # Count objects per selected class
    total_pixels = mask_full.size
    min_area = 25
    _, _, _ = countObject(pred_mask, class_ids=list(selected_set), min_area=min_area)

    # Build formatted reasoning
    lines = [f"The input image was analysed with {model_key}:\n"]
    for class_id in sorted(selected_set):
        class_pixels = int(np.sum(mask_full == class_id))
        if class_pixels == 0:
            continue
        from scipy import ndimage as ndi
        region_mask = (mask_full == class_id).astype(np.uint8)
        labeled, num_blobs = ndi.label(region_mask)
        count = sum(1 for i in range(1, num_blobs + 1) if np.sum(labeled == i) >= min_area)
        pct = class_pixels / total_pixels * 100
        lines.append(f"+ {count} {CLASSES[class_id]}(s) found! ({pct:.1f}%)")

    return jsonify({
        "segmentation_image": seg_b64,
        "reasoning": "\n".join(lines)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
