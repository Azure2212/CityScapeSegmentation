"""Urban-scene analysis helpers for CityScapes segmentation masks.

This module stays downstream of the segmentation model. It interprets a
predicted semantic mask as an explainable scene-parsing layer for the urban
dashboard and report.
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
REGION_CLASS_IDS = (6, 7, 11, 12, 13, 14, 15, 16, 17, 18)
STUFF_CONTEXT_CLASS_IDS = (0, 1, 2, 3, 4, 8, 9, 10)
BUILT_CLASS_IDS = (2, 3, 4, 5, 6, 7)
GREEN_CLASS_IDS = (8, 9)

VERTICAL_BANDS = (
    ("top", "Top", 0.0, 1.0 / 3.0),
    ("middle", "Middle", 1.0 / 3.0, 2.0 / 3.0),
    ("bottom", "Bottom", 2.0 / 3.0, 1.0),
)
HORIZONTAL_BANDS = (
    ("left", "Left", 0.0, 1.0 / 3.0),
    ("center", "Center", 1.0 / 3.0, 2.0 / 3.0),
    ("right", "Right", 2.0 / 3.0, 1.0),
)

SCENE_TAG_LABELS = {
    "vehicle_dominant_corridor": "Vehicle-dominant corridor",
    "pedestrian_supportive_street": "Pedestrian-supportive street",
    "green_corridor": "Green corridor",
    "dense_built_frontage": "Dense built frontage",
    "open_view_street": "Open-view street",
    "mixed_mobility_scene": "Mixed-mobility scene",
}


def mask_to_array(mask: Any) -> np.ndarray:
    """Convert torch/numpy-like masks into a 2D integer NumPy array."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask, got shape {array.shape}")

    return array.astype(np.int64, copy=False)


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _percentage(pixel_count: int, total_pixels: int) -> float:
    if total_pixels == 0:
        return 0.0
    return round((pixel_count / total_pixels) * 100.0, 2)


def _score_item(key: str, label: str, score: float, detail: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "score": round(max(0.0, min(100.0, score)), 1),
        "detail": detail,
    }


def _relation_item(key: str, label: str, active: bool, count: int, detail: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "active": bool(active),
        "count": int(count),
        "detail": detail,
    }


def _warning_item(key: str, detail: str) -> dict[str, Any]:
    return {"key": key, "detail": detail}


def _top_visible_classes(mask_array: np.ndarray, classes: dict[int, str], top_n: int = 3) -> list[dict[str, Any]]:
    total_pixels = int(mask_array.size)
    max_class_id = max(classes)
    pixel_counts = np.bincount(mask_array.ravel(), minlength=max_class_id + 1)
    visible = []
    for class_id in sorted(classes):
        pixels = int(pixel_counts[class_id]) if class_id < len(pixel_counts) else 0
        if pixels <= 0:
            continue
        visible.append(
            {
                "id": class_id,
                "label": classes[class_id],
                "pixels": pixels,
                "percentage": _percentage(pixels, total_pixels),
            }
        )

    return sorted(visible, key=lambda item: item["pixels"], reverse=True)[:top_n]


def _value_by_id(items: list[dict[str, Any]], key_field: str, value_field: str) -> dict[Any, Any]:
    return {item[key_field]: item[value_field] for item in items}


def _bounds(length: int, specs: tuple[tuple[str, str, float, float], ...]) -> list[tuple[str, str, int, int]]:
    bounds = []
    for index, (key, label, start_ratio, end_ratio) in enumerate(specs):
        start = int(round(start_ratio * length))
        end = int(round(end_ratio * length)) if index < len(specs) - 1 else length
        end = max(start + 1, min(end, length))
        bounds.append((key, label, start, end))
    return bounds


def _slice_band(mask_array: np.ndarray, axis: int, start: int, end: int) -> np.ndarray:
    if axis == 0:
        return mask_array[start:end, :]
    return mask_array[:, start:end]


def _band_percentages(mask_array: np.ndarray, classes: dict[int, str]) -> dict[int, float]:
    total_pixels = int(mask_array.size)
    max_class_id = max(classes)
    pixel_counts = np.bincount(mask_array.ravel(), minlength=max_class_id + 1)
    return {
        class_id: _percentage(int(pixel_counts[class_id]), total_pixels) if class_id < len(pixel_counts) else 0.0
        for class_id in classes
    }


def _build_band_profiles(
    mask_array: np.ndarray,
    classes: dict[int, str],
    axis: int,
    specs: tuple[tuple[str, str, float, float], ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[int, float]]]:
    profiles: list[dict[str, Any]] = []
    percentage_lookup: dict[str, dict[int, float]] = {}

    length = mask_array.shape[0] if axis == 0 else mask_array.shape[1]
    for key, label, start, end in _bounds(length, specs):
        band_mask = _slice_band(mask_array, axis=axis, start=start, end=end)
        band_top_classes = _top_visible_classes(band_mask, classes=classes, top_n=3)
        band_percentages = _band_percentages(band_mask, classes=classes)
        dominant = band_top_classes[0] if band_top_classes else None

        profiles.append(
            {
                "key": key,
                "label": label,
                "start": int(start),
                "end": int(end),
                "dominant_class_id": None if dominant is None else dominant["id"],
                "dominant_label": None if dominant is None else dominant["label"],
                "dominant_percentage": 0.0 if dominant is None else dominant["percentage"],
                "top_classes": band_top_classes,
            }
        )
        percentage_lookup[key] = band_percentages

    return profiles, percentage_lookup


def _score_level(score: float) -> str:
    if score >= 67.0:
        return "high"
    if score >= 34.0:
        return "moderate"
    return "low"


def _band_from_position(value: float, length: int, specs: tuple[tuple[str, str, float, float], ...]) -> str:
    ratio = 0.0 if length <= 0 else value / max(length, 1)
    for key, _, start_ratio, end_ratio in specs:
        if ratio < end_ratio or key == specs[-1][0]:
            if ratio >= start_ratio:
                return key
    return specs[-1][0]


def _adjacent_context(
    mask_array: np.ndarray,
    component: np.ndarray,
    classes: dict[int, str],
    proximity_radius: int,
) -> list[dict[str, Any]]:
    shell = ndimage.binary_dilation(component, iterations=proximity_radius) & ~component
    if not np.any(shell):
        return []

    labels, counts = np.unique(mask_array[shell], return_counts=True)
    total = int(counts.sum())
    adjacent = []
    for label, count in sorted(zip(labels.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True):
        if int(label) not in STUFF_CONTEXT_CLASS_IDS:
            continue
        adjacent.append(
            {
                "class_id": int(label),
                "label": classes.get(int(label), f"class_{label}"),
                "pixels": int(count),
                "percentage": _percentage(int(count), total),
            }
        )
    return adjacent[:3]


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

    component_areas = np.bincount(labeled.ravel())[1:]
    return int(np.count_nonzero(component_areas >= min_area))


def compute_layout_profile(
    mask: Any,
    classes: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Describe whole-scene layout using vertical and horizontal bands."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)

    vertical_bands, vertical_pct = _build_band_profiles(mask_array, classes, axis=0, specs=VERTICAL_BANDS)
    horizontal_bands, horizontal_pct = _build_band_profiles(mask_array, classes, axis=1, specs=HORIZONTAL_BANDS)

    top_sky = vertical_pct["top"].get(10, 0.0)
    middle_sky = vertical_pct["middle"].get(10, 0.0)
    bottom_sky = vertical_pct["bottom"].get(10, 0.0)
    top_road = vertical_pct["top"].get(0, 0.0)
    bottom_road = vertical_pct["bottom"].get(0, 0.0)
    middle_road = vertical_pct["middle"].get(0, 0.0)
    left_built = sum(horizontal_pct["left"].get(class_id, 0.0) for class_id in BUILT_CLASS_IDS)
    right_built = sum(horizontal_pct["right"].get(class_id, 0.0) for class_id in BUILT_CLASS_IDS)
    left_green = sum(horizontal_pct["left"].get(class_id, 0.0) for class_id in GREEN_CLASS_IDS)
    right_green = sum(horizontal_pct["right"].get(class_id, 0.0) for class_id in GREEN_CLASS_IDS)
    left_sidewalk = horizontal_pct["left"].get(1, 0.0)
    right_sidewalk = horizontal_pct["right"].get(1, 0.0)

    center_mobility = horizontal_pct["center"].get(0, 0.0) + horizontal_pct["center"].get(1, 0.0)
    center_green = sum(horizontal_pct["center"].get(class_id, 0.0) for class_id in GREEN_CLASS_IDS)
    center_built = sum(horizontal_pct["center"].get(class_id, 0.0) for class_id in BUILT_CLASS_IDS)
    center_sky = horizontal_pct["center"].get(10, 0.0)
    center_corridor_score = max(0.0, min(100.0, center_mobility + (0.5 * center_sky) + (0.25 * center_green) - (0.5 * center_built)))

    sidewalk_gap = abs(left_sidewalk - right_sidewalk)
    sidewalk_balance_score = max(0.0, 100.0 - min(100.0, sidewalk_gap * 5.0))
    edge_green_score = max(left_green, right_green)
    edge_built_score = max(left_built, right_built)

    priors = {
        "sky_is_top": {
            "key": "sky_is_top",
            "label": "Sky concentrated at top",
            "active": bool(top_sky >= 10.0 and top_sky > middle_sky and top_sky > bottom_sky and top_sky > top_road),
            "score": _round(top_sky, 1),
            "detail": f"Sky covers {top_sky:.1f}% of the top band.",
        },
        "road_is_bottom": {
            "key": "road_is_bottom",
            "label": "Road concentrated at bottom",
            "active": bool(bottom_road >= 15.0 and bottom_road > middle_road and bottom_road > top_road),
            "score": _round(bottom_road, 1),
            "detail": f"Road covers {bottom_road:.1f}% of the bottom band.",
        },
        "building_edge_presence": {
            "key": "building_edge_presence",
            "label": "Built edges present",
            "active": bool(edge_built_score >= 18.0),
            "score": _round(edge_built_score, 1),
            "detail": f"Built classes cover {left_built:.1f}% on the left edge and {right_built:.1f}% on the right edge.",
        },
        "left_right_sidewalk_balance": {
            "key": "left_right_sidewalk_balance",
            "label": "Left/right sidewalk balance",
            "active": bool(left_sidewalk > 0.0 and right_sidewalk > 0.0 and sidewalk_gap <= 12.0),
            "score": _round(sidewalk_balance_score, 1),
            "detail": f"Sidewalk share differs by {sidewalk_gap:.1f} percentage points between left and right bands.",
        },
        "green_edge_presence": {
            "key": "green_edge_presence",
            "label": "Green edges present",
            "active": bool(edge_green_score >= 10.0),
            "score": _round(edge_green_score, 1),
            "detail": f"Vegetation/terrain reaches {left_green:.1f}% on the left edge and {right_green:.1f}% on the right edge.",
        },
        "center_corridor_openness": {
            "key": "center_corridor_openness",
            "label": "Center corridor openness",
            "active": bool(center_corridor_score >= 35.0),
            "score": _round(center_corridor_score, 1),
            "detail": f"The center band combines {center_mobility:.1f}% mobility surface, {center_sky:.1f}% sky, and {center_built:.1f}% built coverage.",
        },
    }

    if priors["sky_is_top"]["active"] and priors["road_is_bottom"]["active"] and priors["building_edge_presence"]["active"]:
        dominant_layout = "structured street corridor"
    elif priors["green_edge_presence"]["active"] and priors["center_corridor_openness"]["active"]:
        dominant_layout = "open green corridor"
    elif priors["sky_is_top"]["active"] and priors["road_is_bottom"]["active"]:
        dominant_layout = "typical street composition"
    else:
        dominant_layout = "mixed urban composition"

    return {
        "dominant_layout": dominant_layout,
        "vertical_bands": vertical_bands,
        "horizontal_bands": horizontal_bands,
        "priors": priors,
    }


def compute_region_stats(
    mask: Any,
    classes: dict[int, str] | None = None,
    min_area: int = 25,
    proximity_radius: int = 4,
) -> dict[str, Any]:
    """Build approximate thing/stuff region descriptors from connected components."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)
    total_pixels = int(mask_array.size)
    height, width = mask_array.shape

    class_groups = []
    total_regions = 0

    for class_id in REGION_CLASS_IDS:
        labeled, num_labels = ndimage.label(mask_array == class_id)
        regions = []

        for label_id in range(1, num_labels + 1):
            component = labeled == label_id
            area = int(np.count_nonzero(component))
            if area < min_area:
                continue

            ys, xs = np.nonzero(component)
            centroid_y = float(np.mean(ys))
            centroid_x = float(np.mean(xs))
            adjacent_stuff_classes = _adjacent_context(mask_array, component, classes, proximity_radius=proximity_radius)
            vertical_band = _band_from_position(centroid_y, height, VERTICAL_BANDS)
            horizontal_band = _band_from_position(centroid_x, width, HORIZONTAL_BANDS)

            regions.append(
                {
                    "region_id": f"{class_id}-{len(regions) + 1}",
                    "area": area,
                    "percentage": _percentage(area, total_pixels),
                    "centroid": {"x": _round(centroid_x, 1), "y": _round(centroid_y, 1)},
                    "bbox": {
                        "x1": int(xs.min()),
                        "y1": int(ys.min()),
                        "x2": int(xs.max()),
                        "y2": int(ys.max()),
                    },
                    "band_location": {
                        "vertical": vertical_band,
                        "horizontal": horizontal_band,
                    },
                    "adjacent_stuff_classes": adjacent_stuff_classes,
                }
            )

        if not regions:
            continue

        total_regions += len(regions)
        vertical_counts = {}
        horizontal_counts = {}
        for region in regions:
            vertical_counts[region["band_location"]["vertical"]] = vertical_counts.get(region["band_location"]["vertical"], 0) + 1
            horizontal_counts[region["band_location"]["horizontal"]] = horizontal_counts.get(region["band_location"]["horizontal"], 0) + 1

        class_groups.append(
            {
                "class_id": class_id,
                "label": classes[class_id],
                "count": len(regions),
                "total_pixels": int(sum(region["area"] for region in regions)),
                "percentage": _percentage(sum(region["area"] for region in regions), total_pixels),
                "dominant_vertical_band": max(vertical_counts, key=vertical_counts.get),
                "dominant_horizontal_band": max(horizontal_counts, key=horizontal_counts.get),
                "regions": regions,
            }
        )

    class_groups.sort(key=lambda item: (item["count"], item["total_pixels"]), reverse=True)

    return {
        "total_regions": total_regions,
        "classes": class_groups,
    }


def compute_object_counts(
    mask: Any,
    classes: dict[int, str] | None = None,
    min_area: int = 25,
    region_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Count countable urban objects from sufficiently large mask blobs."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)
    total_pixels = int(mask_array.size)
    class_pixels = _value_by_id(compute_class_stats(mask_array, classes=classes), "id", "pixels")
    region_counts = {}

    if region_stats is not None:
        for item in region_stats.get("classes", []):
            region_counts[item["class_id"]] = int(item["count"])

    counts = []
    for class_id in COUNTABLE_CLASS_IDS:
        count = region_counts.get(class_id)
        if count is None:
            count = count_connected_components(mask_array, class_id=class_id, min_area=min_area)
        pixels = int(class_pixels.get(class_id, 0))
        counts.append(
            {
                "class_id": class_id,
                "label": classes[class_id],
                "count": int(count),
                "pixels": pixels,
                "percentage": _percentage(pixels, total_pixels),
            }
        )

    return counts


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
    built_pct = sum(pct.get(class_id, 0.0) for class_id in BUILT_CLASS_IDS)
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
    """Estimate whether active-mobility regions sit closer to road or sidewalk."""
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
        signal = "road-adjacent active mobility"
    elif sidewalk_adjacent > 0:
        signal = "sidewalk-adjacent active mobility"
    else:
        signal = "no strong active-mobility context"

    return {
        "road_adjacent_active_mobility": road_adjacent,
        "sidewalk_adjacent_active_mobility": sidewalk_adjacent,
        "unclassified_active_mobility": unclassified,
        "signal": signal,
        "note": "Adjacency is estimated from nearby segmentation labels, not measured scene geometry.",
    }


def _has_adjacent_label(region: dict[str, Any], label: str) -> bool:
    return any(item["label"] == label for item in region.get("adjacent_stuff_classes", []))


def compute_relation_flags(
    mask: Any,
    region_stats: dict[str, Any],
    layout_profile: dict[str, Any],
    classes: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Turn region and layout structure into explainable scene relations."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)

    regions_by_class = {item["class_id"]: item["regions"] for item in region_stats.get("classes", [])}
    active_regions = regions_by_class.get(11, []) + regions_by_class.get(12, [])
    bicycle_regions = regions_by_class.get(18, [])
    vehicle_regions = []
    for class_id in MOTORIZED_CLASS_IDS:
        vehicle_regions.extend(regions_by_class.get(class_id, []))

    road_pixels = np.argwhere(mask_array == 0)
    road_centroid_y = float(np.mean(road_pixels[:, 0])) if road_pixels.size else None

    person_on_sidewalk = sum(1 for region in active_regions if _has_adjacent_label(region, "sidewalk"))
    person_near_road = sum(1 for region in active_regions if _has_adjacent_label(region, "road"))
    bike_near_sidewalk = sum(1 for region in bicycle_regions if _has_adjacent_label(region, "sidewalk"))
    vehicle_in_road_corridor = sum(
        1
        for region in vehicle_regions
        if region["band_location"]["horizontal"] == "center" and _has_adjacent_label(region, "road")
    )
    traffic_sign_near_road = sum(1 for region in regions_by_class.get(7, []) if _has_adjacent_label(region, "road"))
    traffic_light_above_road = sum(
        1
        for region in regions_by_class.get(6, [])
        if road_centroid_y is not None and region["centroid"]["y"] < road_centroid_y and _has_adjacent_label(region, "road")
    )

    priors = layout_profile["priors"]
    horizontal_bands = {band["key"]: band for band in layout_profile["horizontal_bands"]}
    green_edges = sum(
        1
        for band_key in ("left", "right")
        if any(
            item["id"] in GREEN_CLASS_IDS
            for item in horizontal_bands.get(band_key, {}).get("top_classes", [])
        )
    )
    left_right_built = priors["building_edge_presence"]["score"]
    greenery_active = priors["green_edge_presence"]["active"]
    building_continuity_active = bool(
        priors["building_edge_presence"]["active"]
        and priors["left_right_sidewalk_balance"]["score"] >= 30.0
    )

    return {
        "person_on_sidewalk": _relation_item(
            "person_on_sidewalk",
            "People on sidewalk context",
            person_on_sidewalk > 0,
            person_on_sidewalk,
            f"{person_on_sidewalk} person/rider region(s) sit adjacent to sidewalk labels.",
        ),
        "person_near_road": _relation_item(
            "person_near_road",
            "People near road",
            person_near_road > 0,
            person_near_road,
            f"{person_near_road} person/rider region(s) sit adjacent to road labels.",
        ),
        "bike_near_sidewalk": _relation_item(
            "bike_near_sidewalk",
            "Bicycles near sidewalk",
            bike_near_sidewalk > 0,
            bike_near_sidewalk,
            f"{bike_near_sidewalk} bicycle region(s) sit adjacent to sidewalk labels.",
        ),
        "vehicle_in_road_corridor": _relation_item(
            "vehicle_in_road_corridor",
            "Vehicles in road corridor",
            vehicle_in_road_corridor > 0,
            vehicle_in_road_corridor,
            f"{vehicle_in_road_corridor} motor-vehicle region(s) occupy the center road corridor.",
        ),
        "traffic_sign_near_road": _relation_item(
            "traffic_sign_near_road",
            "Traffic signs near road",
            traffic_sign_near_road > 0,
            traffic_sign_near_road,
            f"{traffic_sign_near_road} traffic-sign region(s) sit next to road labels.",
        ),
        "traffic_light_above_road": _relation_item(
            "traffic_light_above_road",
            "Traffic lights above road",
            traffic_light_above_road > 0,
            traffic_light_above_road,
            f"{traffic_light_above_road} traffic-light region(s) sit above the road centroid and near road labels.",
        ),
        "greenery_along_edge": _relation_item(
            "greenery_along_edge",
            "Greenery along edges",
            greenery_active,
            green_edges if greenery_active else 0,
            priors["green_edge_presence"]["detail"],
        ),
        "building_frontage_continuity": _relation_item(
            "building_frontage_continuity",
            "Building frontage continuity",
            building_continuity_active,
            int(left_right_built >= 18.0),
            f"Built-edge presence scores {left_right_built:.1f} with left/right sidewalk balance {priors['left_right_sidewalk_balance']['score']:.1f}.",
        ),
    }


def compute_scene_tags(
    class_stats: list[dict[str, Any]],
    group_stats: list[dict[str, Any]],
    object_counts: list[dict[str, Any]],
    planning_scores: dict[str, dict[str, Any]],
    layout_profile: dict[str, Any],
    relation_flags: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create interpretable scene tags with exact evidence."""
    pct = {item["id"]: float(item["percentage"]) for item in class_stats}
    group_pct = {item["key"]: float(item["percentage"]) for item in group_stats}
    count_by_label = {item["label"]: int(item["count"]) for item in object_counts}

    vehicle_count = sum(count_by_label.get(label, 0) for label in ("car", "truck", "bus", "train", "motorcycle"))
    active_count = count_by_label.get("person", 0) + count_by_label.get("rider", 0) + count_by_label.get("bicycle", 0)
    tags = []

    road_score = planning_scores["road_car_dominance"]["score"]
    pedestrian_score = planning_scores["pedestrian_support"]["score"]
    green_score = planning_scores["green_space_share"]["score"]
    built_score = planning_scores["built_density"]["score"]
    open_score = planning_scores["open_view_signal"]["score"]
    corridor_score = layout_profile["priors"]["center_corridor_openness"]["score"]

    if road_score >= 55.0 or relation_flags["vehicle_in_road_corridor"]["count"] >= 2:
        tags.append(
            {
                "key": "vehicle_dominant_corridor",
                "label": SCENE_TAG_LABELS["vehicle_dominant_corridor"],
                "detail": "Road and motor traffic dominate the center corridor.",
                "evidence": [
                    f"Road/car dominance score {road_score:.1f}.",
                    f"{relation_flags['vehicle_in_road_corridor']['count']} vehicle region(s) occupy the center road corridor.",
                    f"Mobility surface covers {group_pct.get('mobility_surface', 0.0):.1f}% of the scene.",
                ],
            }
        )

    if pedestrian_score >= 45.0 and (
        relation_flags["person_on_sidewalk"]["active"] or relation_flags["bike_near_sidewalk"]["active"]
    ):
        tags.append(
            {
                "key": "pedestrian_supportive_street",
                "label": SCENE_TAG_LABELS["pedestrian_supportive_street"],
                "detail": "Sidewalk share and active-use context suggest a pedestrian-friendly street edge.",
                "evidence": [
                    f"Pedestrian-support score {pedestrian_score:.1f}.",
                    relation_flags["person_on_sidewalk"]["detail"],
                    relation_flags["bike_near_sidewalk"]["detail"],
                ],
            }
        )

    if green_score >= 15.0 and relation_flags["greenery_along_edge"]["active"]:
        tags.append(
            {
                "key": "green_corridor",
                "label": SCENE_TAG_LABELS["green_corridor"],
                "detail": "Green/open classes visibly frame the street corridor.",
                "evidence": [
                    f"Green-space share {green_score:.1f}%.",
                    relation_flags["greenery_along_edge"]["detail"],
                    f"Open-view score {open_score:.1f}.",
                ],
            }
        )

    if built_score >= 20.0 and relation_flags["building_frontage_continuity"]["active"]:
        tags.append(
            {
                "key": "dense_built_frontage",
                "label": SCENE_TAG_LABELS["dense_built_frontage"],
                "detail": "Built form strongly frames the left and right street edges.",
                "evidence": [
                    f"Built-density score {built_score:.1f}.",
                    relation_flags["building_frontage_continuity"]["detail"],
                    layout_profile["priors"]["building_edge_presence"]["detail"],
                ],
            }
        )

    if open_score >= 25.0 and layout_profile["priors"]["sky_is_top"]["active"] and corridor_score >= 35.0:
        tags.append(
            {
                "key": "open_view_street",
                "label": SCENE_TAG_LABELS["open_view_street"],
                "detail": "The scene keeps a visible sky channel above the street corridor.",
                "evidence": [
                    f"Open-view score {open_score:.1f}.",
                    layout_profile["priors"]["sky_is_top"]["detail"],
                    layout_profile["priors"]["center_corridor_openness"]["detail"],
                ],
            }
        )

    if vehicle_count > 0 and active_count > 0:
        tags.append(
            {
                "key": "mixed_mobility_scene",
                "label": SCENE_TAG_LABELS["mixed_mobility_scene"],
                "detail": "Motorized and active-mobility regions coexist in the same scene.",
                "evidence": [
                    f"{vehicle_count} motor-vehicle region(s) detected.",
                    f"{active_count} active-mobility region(s) detected.",
                    f"Active-mobility score {planning_scores['active_mobility']['score']:.1f}.",
                ],
            }
        )

    return tags


def compute_analysis_warnings(
    class_stats: list[dict[str, Any]],
    group_stats: list[dict[str, Any]],
    region_stats: dict[str, Any],
    layout_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flag cases where the downstream interpretation should stay cautious."""
    warnings = []
    visible_classes = [item for item in class_stats if item["pixels"] > 0]
    group_pct = {item["key"]: float(item["percentage"]) for item in group_stats}

    if len(visible_classes) <= 2:
        warnings.append(
            _warning_item(
                "low_semantic_diversity",
                f"Only {len(visible_classes)} class(es) are visible, so scene tags may be under-informed.",
            )
        )

    if group_pct.get("mobility_surface", 0.0) < 5.0:
        warnings.append(
            _warning_item(
                "weak_street_surface",
                f"Mobility surfaces cover only {group_pct.get('mobility_surface', 0.0):.1f}% of the image.",
            )
        )

    if not layout_profile["priors"]["sky_is_top"]["active"] and not layout_profile["priors"]["road_is_bottom"]["active"]:
        warnings.append(
            _warning_item(
                "weak_layout_prior",
                "The image does not strongly match the expected sky-over-road street layout.",
            )
        )

    if region_stats.get("total_regions", 0) == 0:
        warnings.append(
            _warning_item(
                "no_region_structure",
                "No countable regions survived the area threshold, so relation analysis is sparse.",
            )
        )

    return warnings


def build_planning_summary(
    class_stats: list[dict[str, Any]],
    group_stats: list[dict[str, Any]],
    object_counts: list[dict[str, Any]],
    planning_scores: dict[str, dict[str, Any]],
    spatial_flags: dict[str, Any],
    layout_profile: dict[str, Any],
    scene_tags: list[dict[str, Any]],
) -> str:
    """Build a concise interpretation grounded in the mask output."""
    visible_classes = [item for item in class_stats if item["pixels"] > 0]
    top_classes = sorted(visible_classes, key=lambda item: item["percentage"], reverse=True)[:3]
    top_text = ", ".join(f"{item['label']} ({item['percentage']:.1f}%)" for item in top_classes)

    group_by_key = {item["key"]: item for item in group_stats}
    count_by_label = {item["label"]: item["count"] for item in object_counts}
    vehicle_count = sum(count_by_label.get(label, 0) for label in ("car", "truck", "bus", "train", "motorcycle"))
    active_count = count_by_label.get("person", 0) + count_by_label.get("rider", 0) + count_by_label.get("bicycle", 0)
    tag_text = ", ".join(tag["label"].lower() for tag in scene_tags[:3]) if scene_tags else "no strong scene tags"

    green_score = planning_scores["green_space_share"]["score"]
    road_score = planning_scores["road_car_dominance"]["score"]
    pedestrian_score = planning_scores["pedestrian_support"]["score"]
    built_score = planning_scores["built_density"]["score"]

    return (
        f"The dominant predicted classes are {top_text}. "
        f"The layout reads as {layout_profile['dominant_layout']}. "
        f"Green/open coverage is {group_by_key['green_open_space']['percentage']:.1f}% "
        f"with a {_score_level(green_score)} green-space signal. "
        f"Mobility surfaces cover {group_by_key['mobility_surface']['percentage']:.1f}% "
        f"and the road/car dominance signal is {_score_level(road_score)}. "
        f"The scene includes {vehicle_count} motor-vehicle region(s) and {active_count} "
        f"people/active-mobility region(s), giving a {_score_level(pedestrian_score)} "
        f"pedestrian-support signal. Built-density reads as {_score_level(built_score)}. "
        f"Scene tags: {tag_text}. "
        f"Spatial context: {spatial_flags['signal']}. "
        "These are segmentation-derived planning estimates, not surveyed ground truth."
    )


def compare_scene_analyses(
    analyses: dict[str, dict[str, Any]],
    model_labels: dict[str, str] | None = None,
    skipped_models: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compare scene-analysis outputs across multiple segmentation models."""
    model_labels = model_labels or {}
    skipped_models = skipped_models or []
    model_keys = list(analyses.keys())

    if not model_keys:
        return {
            "models": [],
            "shared_tags": [],
            "class_delta_highlights": [],
            "group_delta_highlights": [],
            "disagreement_notes": ["No model analyses were available for comparison."],
            "skipped_models": skipped_models,
        }

    tag_sets = {}
    tag_lookup = {}
    for model_key in model_keys:
        tags = analyses[model_key].get("scene_tags", [])
        tag_sets[model_key] = {tag["key"] for tag in tags}
        for tag in tags:
            tag_lookup.setdefault(tag["key"], tag["label"])

    shared_tag_keys = set.intersection(*(tag_sets[key] for key in model_keys)) if len(model_keys) > 1 else tag_sets[model_keys[0]]
    union_tag_keys = set().union(*(tag_sets[key] for key in model_keys))

    class_highlights = []
    for class_id, label in DEFAULT_CLASSES.items():
        values = []
        for model_key in model_keys:
            class_map = _value_by_id(analyses[model_key]["class_stats"], "id", "percentage")
            values.append((model_key, float(class_map.get(class_id, 0.0))))
        min_model, min_value = min(values, key=lambda item: item[1])
        max_model, max_value = max(values, key=lambda item: item[1])
        diff = max_value - min_value
        if diff < 5.0:
            continue
        class_highlights.append(
            {
                "class_id": class_id,
                "label": label,
                "range": _round(diff, 2),
                "min_model": min_model,
                "min_label": model_labels.get(min_model, min_model),
                "min_percentage": _round(min_value, 2),
                "max_model": max_model,
                "max_label": model_labels.get(max_model, max_model),
                "max_percentage": _round(max_value, 2),
            }
        )
    class_highlights.sort(key=lambda item: item["range"], reverse=True)

    group_highlights = []
    for group in PLANNING_GROUPS:
        values = []
        for model_key in model_keys:
            group_map = _value_by_id(analyses[model_key]["group_stats"], "key", "percentage")
            values.append((model_key, float(group_map.get(group["key"], 0.0))))
        min_model, min_value = min(values, key=lambda item: item[1])
        max_model, max_value = max(values, key=lambda item: item[1])
        diff = max_value - min_value
        if diff < 5.0:
            continue
        group_highlights.append(
            {
                "key": group["key"],
                "label": group["label"],
                "range": _round(diff, 2),
                "min_model": min_model,
                "min_label": model_labels.get(min_model, min_model),
                "min_percentage": _round(min_value, 2),
                "max_model": max_model,
                "max_label": model_labels.get(max_model, max_model),
                "max_percentage": _round(max_value, 2),
            }
        )
    group_highlights.sort(key=lambda item: item["range"], reverse=True)

    disagreement_notes = []
    if shared_tag_keys:
        shared_labels = ", ".join(tag_lookup[key] for key in sorted(shared_tag_keys))
        disagreement_notes.append(f"All compared models agree on these scene tags: {shared_labels}.")
    elif len(model_keys) > 1:
        disagreement_notes.append("The compared models do not share a single scene tag intersection.")

    if class_highlights:
        item = class_highlights[0]
        disagreement_notes.append(
            f"The largest class spread is {item['label']}: {item['min_label']} sees {item['min_percentage']:.1f}% while {item['max_label']} sees {item['max_percentage']:.1f}%."
        )

    if group_highlights:
        item = group_highlights[0]
        disagreement_notes.append(
            f"The largest group spread is {item['label']}: {item['min_label']} sees {item['min_percentage']:.1f}% while {item['max_label']} sees {item['max_percentage']:.1f}%."
        )

    unique_tag_keys = union_tag_keys - shared_tag_keys
    if unique_tag_keys:
        unique_labels = ", ".join(sorted(tag_lookup[key] for key in unique_tag_keys))
        disagreement_notes.append(f"Only some models emit these tags: {unique_labels}.")

    return {
        "models": [
            {
                "key": model_key,
                "label": model_labels.get(model_key, model_key),
                "dominant_layout": analyses[model_key]["layout_profile"]["dominant_layout"],
                "scene_tags": [tag["label"] for tag in analyses[model_key].get("scene_tags", [])],
                "summary": analyses[model_key]["summary"],
            }
            for model_key in model_keys
        ],
        "shared_tags": [
            {"key": key, "label": tag_lookup[key]}
            for key in sorted(shared_tag_keys)
        ],
        "class_delta_highlights": class_highlights[:5],
        "group_delta_highlights": group_highlights[:5],
        "disagreement_notes": disagreement_notes,
        "skipped_models": skipped_models,
    }


def analyze_urban_scene(
    mask: Any,
    classes: dict[int, str] | None = None,
    min_area: int = 25,
) -> dict[str, Any]:
    """Run the full urban scene-analysis pipeline for one predicted mask."""
    classes = classes or DEFAULT_CLASSES
    mask_array = mask_to_array(mask)
    total_pixels = int(mask_array.size)

    class_stats = compute_class_stats(mask_array, classes=classes)
    group_stats = compute_group_stats(class_stats, total_pixels=total_pixels)
    layout_profile = compute_layout_profile(mask_array, classes=classes)
    region_stats = compute_region_stats(mask_array, classes=classes, min_area=min_area)
    object_counts = compute_object_counts(mask_array, classes=classes, min_area=min_area, region_stats=region_stats)
    planning_scores = compute_planning_scores(class_stats, object_counts)
    spatial_flags = compute_spatial_flags(mask_array, min_area=min_area)
    relation_flags = compute_relation_flags(mask_array, region_stats=region_stats, layout_profile=layout_profile, classes=classes)
    scene_tags = compute_scene_tags(
        class_stats=class_stats,
        group_stats=group_stats,
        object_counts=object_counts,
        planning_scores=planning_scores,
        layout_profile=layout_profile,
        relation_flags=relation_flags,
    )
    analysis_warnings = compute_analysis_warnings(
        class_stats=class_stats,
        group_stats=group_stats,
        region_stats=region_stats,
        layout_profile=layout_profile,
    )
    summary = build_planning_summary(
        class_stats=class_stats,
        group_stats=group_stats,
        object_counts=object_counts,
        planning_scores=planning_scores,
        spatial_flags=spatial_flags,
        layout_profile=layout_profile,
        scene_tags=scene_tags,
    )

    return {
        "total_pixels": total_pixels,
        "class_stats": class_stats,
        "group_stats": group_stats,
        "object_counts": object_counts,
        "planning_scores": planning_scores,
        "spatial_flags": spatial_flags,
        "layout_profile": layout_profile,
        "region_stats": region_stats,
        "relation_flags": relation_flags,
        "scene_tags": scene_tags,
        "analysis_warnings": analysis_warnings,
        "summary": summary,
    }
