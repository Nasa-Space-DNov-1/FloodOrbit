import numpy as np
from scipy.ndimage import label
from shapely.geometry import box, mapping
from shapely.ops import unary_union

def dual_rule_classify(diff_dB: np.ndarray, post_dB: np.ndarray) -> np.ndarray:
    classification = np.zeros(diff_dB.shape, dtype=np.int8)
    water_mask = (diff_dB <= -3.0) & (post_dB <= -15.0)
    classification[water_mask] = 1
    debris_mask = (diff_dB >= 3.5)
    classification[debris_mask] = 2
    return classification

def apply_mmu_filter(binary_mask: np.ndarray, min_pixel_size: int = 4) -> np.ndarray:
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(binary_mask, structure=structure)
    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    component_sizes = np.bincount(labeled_array.ravel())
    valid_labels = np.where(component_sizes >= min_pixel_size)[0]
    valid_labels = valid_labels[valid_labels != 0]

    filtered_mask = np.isin(labeled_array, valid_labels).astype(np.uint8)
    return filtered_mask

def vectorize_flood_mask_to_geojson(
    cleaned_mask: np.ndarray, origin_lon: float = 74.5, origin_lat: float = 35.8, pixel_deg: float = 0.0002
) -> dict:
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(cleaned_mask, structure=structure)
    features = []

    for cluster_id in range(1, num_features + 1):
        rows, cols = np.where(labeled_array == cluster_id)
        if len(rows) == 0:
            continue

        pixel_boxes = [
            box(origin_lon + c * pixel_deg, origin_lat - (r + 1) * pixel_deg,
                origin_lon + (c + 1) * pixel_deg, origin_lat - r * pixel_deg)
            for r, c in zip(rows, cols)
        ]
        merged_poly = unary_union(pixel_boxes)
        pixel_count = int(len(rows))

        features.append({
            "type": "Feature",
            "properties": {
                "cluster_id": int(cluster_id),
                "pixel_count": pixel_count,
                "area_m2": float(pixel_count * 400.0)
            },
            "geometry": mapping(merged_poly)
        })

    return {"type": "FeatureCollection", "features": features}