"""Urban-planning analysis helpers for CityScapes segmentation masks.

This file is intentionally additive. It does not alter the training or
inference pipeline; it interprets an already-predicted semantic mask as a set
of planning-oriented signals for the companion urban dashboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


DEFAULT_CLASSES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "others",
}


# These groups translate pixel classes into planning concepts. The groups are
# signals rather than a partition: bicycle appears in both vehicle and active
# mobility because it is relevant to both interpretations.
PLANNING_GROUPS = (
    {
        "key": "mobility_surface",
        "label": "Mobility surface",
        "class_ids": (0, 1),
    },
    {
        "key": "built_environment",
        "label": "Built environment",
        "class_ids": (2, 3, 4, 5, 6, 7),
    },
    {
        "key": "green_open_space",
        "label": "Green and open view",
        "class_ids": (8, 9, 10),
    },
    {
        "key": "vehicles",
        "label": "Vehicles",
        "class_ids": (13, 14, 15, 16, 17, 18),
    },
    {
        "key": "people_active_mobility",
        "label": "People and active mobility",
        "class_ids": (11, 12, 18),
    },
)

COUNTABLE_CLASS_IDS = (11, 12, 13, 14, 15, 16, 17, 18)
MOTORIZED_CLASS_IDS = (13, 14, 15, 16, 17)
ACTIVE_MOBILITY_CLASS_IDS = (11, 12, 18)


def mask_to_array(mask: Any) -> np.ndarray:
    """Convert torch/numpy-like masks into a 2D integer NumPy array."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask, got shape {array.shape}")

    return array.astype(np.int64, copy=False)


def _percentage(pixel_count: int, total_pixels: int) -> float:
    if total_pixels == 0:
        return 0.0
    return round((pixel_count / total_pixels) * 100.0, 2)


def compute_class_stats(
    mask: Any,
    classes: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Return per-class pixel coverage for every known CityScapes class."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)
    total_pixels = int(mask_array.size)
    max_class_id = max(classes)
    pixel_counts = np.bincount(mask_array.ravel(), minlength=max_class_id + 1)

    stats = []
    for class_id in sorted(classes):
        pixels = int(pixel_counts[class_id]) if class_id < len(pixel_counts) else 0
        stats.append(
            {
                "id": class_id,
                "label": classes[class_id],
                "pixels": pixels,
                "percentage": _percentage(pixels, total_pixels),
            }
        )

    return stats


def compute_group_stats(
    class_stats: list[dict[str, Any]],
    total_pixels: int,
) -> list[dict[str, Any]]:
    """Aggregate class coverage into planning-oriented groups."""
    pixels_by_class = {item["id"]: item["pixels"] for item in class_stats}
    groups = []

    for group in PLANNING_GROUPS:
        pixels = sum(pixels_by_class.get(class_id, 0) for class_id in group["class_ids"])
        groups.append(
            {
                "key": group["key"],
                "label": group["label"],
                "class_ids": list(group["class_ids"]),
                "pixels": int(pixels),
                "percentage": _percentage(int(pixels), total_pixels),
            }
        )

    return groups


def count_connected_components(mask: Any, class_id: int, min_area: int = 25) -> int:
    """Count separate regions for one class using connected components."""
    mask_array = mask_to_array(mask)
    binary = mask_array == class_id
    labeled, num_labels = ndimage.label(binary)
    if num_labels == 0:
        return 0

    # Label 0 is background; every later bin is one connected component.
    component_areas = np.bincount(labeled.ravel())[1:]
    return int(np.count_nonzero(component_areas >= min_area))


def compute_object_counts(
    mask: Any,
    classes: dict[int, str] | None = None,
    min_area: int = 25,
) -> list[dict[str, Any]]:
    """Count countable urban objects from sufficiently large mask blobs."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)
    total_pixels = int(mask_array.size)

    counts = []
    for class_id in COUNTABLE_CLASS_IDS:
        pixels = int(np.count_nonzero(mask_array == class_id))
        counts.append(
            {
                "class_id": class_id,
                "label": classes[class_id],
                "count": count_connected_components(mask_array, class_id, min_area=min_area),
                "pixels": pixels,
                "percentage": _percentage(pixels, total_pixels),
            }
        )

    return counts


def _score_item(key: str, label: str, score: float, detail: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "score": round(max(0.0, min(100.0, score)), 1),
        "detail": detail,
    }


def compute_planning_scores(
    class_stats: list[dict[str, Any]],
    object_counts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Create transparent heuristic scores from class coverage and counts."""
    pct = {item["id"]: float(item["percentage"]) for item in class_stats}
    count_by_class = {item["class_id"]: int(item["count"]) for item in object_counts}

    road_pct = pct.get(0, 0.0)
    sidewalk_pct = pct.get(1, 0.0)
    green_surface_pct = pct.get(8, 0.0) + pct.get(9, 0.0)
    sky_pct = pct.get(10, 0.0)
    built_pct = sum(pct.get(class_id, 0.0) for class_id in (2, 3, 4, 5, 6, 7))
    motorized_pct = sum(pct.get(class_id, 0.0) for class_id in MOTORIZED_CLASS_IDS)
    active_pct = sum(pct.get(class_id, 0.0) for class_id in ACTIVE_MOBILITY_CLASS_IDS)
    active_count = sum(count_by_class.get(class_id, 0) for class_id in ACTIVE_MOBILITY_CLASS_IDS)

    mobility_surface = road_pct + sidewalk_pct
    sidewalk_balance = (sidewalk_pct / mobility_surface * 100.0) if mobility_surface else 0.0
    active_presence = min(100.0, (active_pct * 8.0) + (active_count * 6.0))

    return {
        "green_space_share": _score_item(
            "green_space_share",
            "Green-space share",
            green_surface_pct,
            "Vegetation and terrain share of the image.",
        ),
        "open_view_signal": _score_item(
            "open_view_signal",
            "Open-view signal",
            green_surface_pct + (sky_pct * 0.5),
            "Green surface plus partial credit for visible sky.",
        ),
        "road_car_dominance": _score_item(
            "road_car_dominance",
            "Road/car dominance",
            road_pct + (motorized_pct * 4.0),
            "Road coverage plus amplified motor-vehicle presence.",
        ),
        "pedestrian_support": _score_item(
            "pedestrian_support",
            "Pedestrian-support signal",
            (sidewalk_balance * 0.75) + (active_presence * 0.25),
            "Sidewalk share within mobility surfaces plus visible active use.",
        ),
        "built_density": _score_item(
            "built_density",
            "Built-density signal",
            built_pct,
            "Buildings, walls, fences, poles, and traffic infrastructure.",
        ),
        "active_mobility": _score_item(
            "active_mobility",
            "Active-mobility signal",
            (active_pct * 10.0) + (active_count * 8.0),
            "People, riders, and bicycles detected as distinct regions.",
        ),
    }


def compute_spatial_flags(
    mask: Any,
    min_area: int = 25,
    proximity_radius: int = 4,
) -> dict[str, Any]:
    """Estimate whether active-mobility regions sit closer to road or sidewalk.

    Semantic masks are mutually exclusive, so a person pixel cannot also be a
    road pixel. This helper uses a small dilated shell around each active region
    and inspects the neighboring labels as a transparent spatial proxy.
    """
    mask_array = mask_to_array(mask)
    road_adjacent = 0
    sidewalk_adjacent = 0
    unclassified = 0

    for class_id in ACTIVE_MOBILITY_CLASS_IDS:
        labeled, num_labels = ndimage.label(mask_array == class_id)
        for label_id in range(1, num_labels + 1):
            component = labeled == label_id
            if int(np.count_nonzero(component)) < min_area:
                continue

            shell = ndimage.binary_dilation(component, iterations=proximity_radius) & ~component
            road_neighbors = int(np.count_nonzero(mask_array[shell] == 0))
            sidewalk_neighbors = int(np.count_nonzero(mask_array[shell] == 1))

            if road_neighbors > sidewalk_neighbors and road_neighbors > 0:
                road_adjacent += 1
            elif sidewalk_neighbors > 0:
                sidewalk_adjacent += 1
            else:
                unclassified += 1

    if road_adjacent > sidewalk_adjacent:
        level = "road-adjacent active mobility"
    elif sidewalk_adjacent > 0:
        level = "sidewalk-adjacent active mobility"
    else:
        level = "no strong active-mobility context"

    return {
        "road_adjacent_active_mobility": road_adjacent,
        "sidewalk_adjacent_active_mobility": sidewalk_adjacent,
        "unclassified_active_mobility": unclassified,
        "signal": level,
        "note": "Adjacency is estimated from nearby segmentation labels, not measured scene geometry.",
    }


def _level(score: float) -> str:
    if score >= 67.0:
        return "high"
    if score >= 34.0:
        return "moderate"
    return "low"


def build_planning_summary(
    class_stats: list[dict[str, Any]],
    group_stats: list[dict[str, Any]],
    object_counts: list[dict[str, Any]],
    planning_scores: dict[str, dict[str, Any]],
    spatial_flags: dict[str, Any],
) -> str:
    """Build a concise interpretation that stays grounded in the mask output."""
    visible_classes = [item for item in class_stats if item["pixels"] > 0]
    top_classes = sorted(visible_classes, key=lambda item: item["percentage"], reverse=True)[:3]
    top_text = ", ".join(f"{item['label']} ({item['percentage']:.1f}%)" for item in top_classes)

    group_by_key = {item["key"]: item for item in group_stats}
    count_by_label = {item["label"]: item["count"] for item in object_counts}
    vehicle_count = sum(count_by_label.get(label, 0) for label in ("car", "truck", "bus", "train", "motorcycle"))
    active_count = count_by_label.get("person", 0) + count_by_label.get("rider", 0) + count_by_label.get("bicycle", 0)

    green_score = planning_scores["green_space_share"]["score"]
    road_score = planning_scores["road_car_dominance"]["score"]
    pedestrian_score = planning_scores["pedestrian_support"]["score"]
    built_score = planning_scores["built_density"]["score"]

    return (
        f"The dominant predicted classes are {top_text}. "
        f"Green/open coverage is {group_by_key['green_open_space']['percentage']:.1f}% "
        f"with a {_level(green_score)} green-space signal. "
        f"Mobility surfaces cover {group_by_key['mobility_surface']['percentage']:.1f}% "
        f"and the road/car dominance signal is {_level(road_score)}. "
        f"The scene includes {vehicle_count} motor-vehicle region(s) and {active_count} "
        f"people/active-mobility region(s), giving a {_level(pedestrian_score)} "
        f"pedestrian-support signal. Built-density reads as {_level(built_score)}. "
        f"Spatial context: {spatial_flags['signal']}. "
        "These are segmentation-derived planning estimates, not surveyed ground truth."
    )


def analyze_urban_scene(
    mask: Any,
    classes: dict[int, str] | None = None,
    min_area: int = 25,
) -> dict[str, Any]:
    """Run the full urban-planning analysis for one predicted mask."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)
    total_pixels = int(mask_array.size)
    class_stats = compute_class_stats(mask_array, classes=classes)
    group_stats = compute_group_stats(class_stats, total_pixels=total_pixels)
    object_counts = compute_object_counts(mask_array, classes=classes, min_area=min_area)
    planning_scores = compute_planning_scores(class_stats, object_counts)
    spatial_flags = compute_spatial_flags(mask_array, min_area=min_area)
    summary = build_planning_summary(
        class_stats=class_stats,
        group_stats=group_stats,
        object_counts=object_counts,
        planning_scores=planning_scores,
        spatial_flags=spatial_flags,
    )

    return {
        "total_pixels": total_pixels,
        "class_stats": class_stats,
        "group_stats": group_stats,
        "object_counts": object_counts,
        "planning_scores": planning_scores,
        "spatial_flags": spatial_flags,
        "summary": summary,
    }
