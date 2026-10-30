import numpy as np
from scipy.ndimage import zoom
from pyproj import CRS, Transformer
import shapely
from shapely.geometry import Point, mapping

def get_local_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)

def create_metric_buffer(lat: float, lon: float, radius_meters: float = 5000.0):
    epsg_code = get_local_utm_epsg(lat, lon)
    utm_crs = CRS.from_epsg(epsg_code)
    wgs84_crs = CRS.from_epsg(4326)

    to_utm = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True).transform

    point_utm = shapely.transform(Point(lon, lat), to_utm)
    buffer_utm = point_utm.buffer(radius_meters)
    buffer_wgs84 = shapely.transform(buffer_utm, to_wgs84)

    geojson_feature = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "utm_epsg": epsg_code,
                "buffer_radius_m": radius_meters
            },
            "geometry": mapping(buffer_wgs84)
        }]
    }
    return geojson_feature, epsg_code

def verify_orbital_compatibility(pre_pass_direction: str, post_pass_direction: str) -> str:
    if pre_pass_direction.upper() == post_pass_direction.upper():
        return "PASSED"
    return "NOT_RECOMMENDED"

def align_raster_grids(pre_grid: np.ndarray, post_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pre_grid.shape != post_grid.shape:
        scale_y = pre_grid.shape[0] / post_grid.shape[0]
        scale_x = pre_grid.shape[1] / post_grid.shape[1]
        post_grid = zoom(post_grid, (scale_y, scale_x), order=1)
    return pre_grid, post_grid

def linear_to_decibel(linear_grid: np.ndarray) -> np.ndarray:
    safe_linear = np.maximum(linear_grid, 1e-6)
    return 10.0 * np.log10(safe_linear)

def compute_decibel_change(pre_linear: np.ndarray, post_linear: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_dB = linear_to_decibel(pre_linear)
    post_dB = linear_to_decibel(post_linear)
    diff_dB = post_dB - pre_dB
    return pre_dB, post_dB, diff_dB