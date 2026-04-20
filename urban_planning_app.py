"""Additive Flask entrypoint for the urban-planning companion dashboard.

Run this file instead of app.py when you want the original app plus the new
urban-planning routes. The existing app object, routes, model registry, model
loading, and original template are imported and reused without modification.
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import torch
from flask import jsonify, render_template, request

from app import MODEL_REGISTRY, app as flask_app, get_model, get_transform
from application import CLASSES, CONFIG_CMAP, DEVICE
from utils.urban_scene_analysis import analyze_urban_scene


def _decode_upload():
    """Decode the uploaded image into OpenCV BGR format."""
    if "image" not in request.files:
        return None, ("No image uploaded", 400)

    file_bytes = np.frombuffer(request.files["image"].read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, ("Cannot decode image", 400)

    return image, None


def _predict_full_mask(image_bgr: np.ndarray, model_key: str) -> np.ndarray:
    """Run the existing model path and resize the mask back to image size."""
    model = get_model(model_key)
    input_size = MODEL_REGISTRY[model_key]["input_size"]
    transform = get_transform(input_size)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = transform(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_mask = torch.argmax(model(tensor), dim=1)

    mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    orig_h, orig_w = image_bgr.shape[:2]
    return cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def _encode_overlay_png(
    image_bgr: np.ndarray,
    mask_full: np.ndarray,
    selected_classes: set[int],
) -> str:
    """Create a translucent class overlay for dashboard visualization only."""
    cmap_colors = (np.array(CONFIG_CMAP.colors) * 255).astype(np.uint8)
    result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    for class_id in selected_classes:
        if 0 <= class_id < len(cmap_colors):
            region = mask_full == class_id
            color = cmap_colors[class_id].astype(np.float32)
            result[region] = (result[region] * 0.38) + (color * 0.62)

    result_bgr = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", result_bgr)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@flask_app.route("/urban")
def urban_dashboard():
    """Render the additive urban-planning dashboard."""
    models = [
        {"key": key, "label": entry["label"], "available": bool(entry["url"])}
        for key, entry in MODEL_REGISTRY.items()
    ]
    return render_template("urban_planning.html", models=models, classes=CLASSES)


@flask_app.route("/urban-predict", methods=["POST"])
def urban_predict():
    """Return segmentation plus urban-planning analysis for one uploaded image."""
    image, error = _decode_upload()
    if error:
        message, status = error
        return jsonify({"error": message}), status

    model_key = request.form.get("model", "UNet")
    if model_key not in MODEL_REGISTRY:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    overlay_selection_sent = request.form.get("overlay_selection_sent") == "1"
    overlay_classes = request.form.getlist("overlay_classes[]", type=int)
    if overlay_selection_sent:
        selected_classes = {class_id for class_id in overlay_classes if class_id in CLASSES}
    else:
        selected_classes = set(CLASSES.keys())

    try:
        mask_full = _predict_full_mask(image, model_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Planning metrics intentionally use the complete predicted mask, regardless
    # of which classes the user chose to visualize in the overlay.
    analysis = analyze_urban_scene(mask_full, classes=CLASSES, min_area=25)
    overlay_b64 = _encode_overlay_png(image, mask_full, selected_classes)

    return jsonify(
        {
            "model": model_key,
            "model_label": MODEL_REGISTRY[model_key]["label"],
            "segmentation_image": overlay_b64,
            **analysis,
        }
    )


# Gunicorn-compatible alias for hosts that expect an ``application`` object.
application = flask_app
app = flask_app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10765))
    flask_app.run(host="0.0.0.0", port=port)
