"""
Road supervision targets derived from vector polygon geometry.

All targets are computed at full image resolution and sliced at patch time,
following the same pattern as build_targets.py.

Target generation is GSD-gated: roads narrower than ~2px at coarse sensors
(SPOT-6, PlanetScope) have no meaningful intra-polygon EDT gradient and are
skipped automatically.
"""

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import MultiPolygon, Polygon

from geotiff_tiler.utils.build_targets import _rasterize_geom

logger = logging.getLogger(__name__)


def _explode_to_polygons(geoms: list) -> list:
    """
    Flatten a mixed list of Polygon / MultiPolygon geometries to individual
    Polygons. Other geometry types are silently dropped.

    Must be called before any code that accesses geom.exterior, since
    MultiPolygon does not have that attribute.
    """
    out = []
    for geom in geoms:
        if isinstance(geom, Polygon):
            if not geom.is_empty:
                out.append(geom)
        elif isinstance(geom, MultiPolygon):
            out.extend(p for p in geom.geoms if not p.is_empty)
    return out


def compute_road_targets(
    road_gdf: gpd.GeoDataFrame,
    image_path: str,
    tmp_dir: str,
    label_name: str,
    max_gsd_for_targets: float = 1.0,
) -> dict[str, str]:
    """
    Compute road supervision targets from polygon geometry.

    Only one target is generated: the intra-polygon EDT (centerline weight).
    Its value is maximum at the polygon centerline and zero at the boundary,
    encoding both cross-section importance and local road width in a single map.

    Target generation is skipped at coarse GSD (> max_gsd_for_targets) where
    roads are 1–2px wide and the intra-polygon EDT has no meaningful gradient.

    Args:
        road_gdf:             GeoDataFrame filtered to road polygons only.
        image_path:           Path to source image (for transform/shape).
        tmp_dir:              Temp directory for output tifs.
        label_name:           Stem used for output filenames.
        max_gsd_for_targets:  GSD threshold in metres above which targets are
                              skipped. Default matches the erosion threshold.

    Returns:
        Dict mapping target name → tif path:
            'roads_centerline_weight'  intra-polygon EDT  uint8
        Returns {} when GSD is too coarse or road_gdf is empty.
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    pixel_size = abs(transform.a)
    if pixel_size > max_gsd_for_targets:
        logger.info(
            f"[road_targets] skipping — GSD {pixel_size:.2f}m > "
            f"threshold {max_gsd_for_targets:.2f}m"
        )
        return {}

    valid_geoms = _explode_to_polygons(
        road_gdf[
            ~road_gdf.geometry.is_empty & road_gdf.geometry.notnull()
        ].geometry.tolist()
    )

    if not valid_geoms:
        return {}

    t = time.time()
    centerline_weight = _compute_road_centerline_weight(valid_geoms, h, w, transform)
    logger.info(f"[road_targets] centerline_weight: {time.time() - t:.1f}s")

    out_path = Path(tmp_dir) / f"{label_name}_roads_centerline_weight.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(centerline_weight, 1)

    logger.info(f"[road_targets] wrote centerline_weight → {out_path}")
    return {"roads_centerline_weight": str(out_path)}


def _compute_road_centerline_weight(
    geoms,
    h: int,
    w: int,
    transform: Affine,
) -> np.ndarray:
    """
    Intra-polygon EDT for road polygons.

    For each road polygon, the distance transform of interior pixels gives
    maximum value at the centerline and zero at the boundary. The value at
    any centerline pixel equals the half-width of the road at that point in
    pixels — so this single map encodes both centerline location and local
    road width.

    Uses the same localised bounding-box approach as the buildings EDT to
    avoid full-image distance transforms. Overlapping polygons are resolved
    by taking the per-pixel maximum so no road is suppressed.

    Normalised to [0, 1] by the global maximum before uint8 storage, making
    the scale consistent across sensors and image extents.

    Args:
        geoms:     List of road polygon geometries.
        h, w:      Image height and width in pixels.
        transform: Rasterio Affine transform.

    Returns:
        (h, w) uint8 array in [0, 255].
    """
    weight = np.zeros((h, w), dtype=np.float32)

    for geom in geoms:
        minx, miny, maxx, maxy = geom.bounds

        # Bounding box in pixel space with 1px padding
        c0 = max(0, int((minx - transform.c) / transform.a) - 1)
        r0 = max(0, int((miny - transform.f) / transform.e) - 1)
        c1 = min(w, int((maxx - transform.c) / transform.a) + 2)
        r1 = min(h, int((maxy - transform.f) / transform.e) + 2)

        local_shape = (r1 - r0, c1 - c0)
        if local_shape[0] <= 0 or local_shape[1] <= 0:
            continue

        local_t = Affine.translation(
            transform.c + c0 * transform.a,
            transform.f + r0 * transform.e,
        ) * Affine.scale(transform.a, transform.e)

        mask = _rasterize_geom(geom, local_shape, local_t)
        if not mask.any():
            continue

        # Intra-polygon EDT: distance from interior pixels to nearest boundary.
        local_edt = distance_transform_edt(mask).astype(np.float32)

        # Max merge: overlapping road polygons don't suppress each other.
        sl = weight[r0:r1, c0:c1]
        np.maximum(sl, local_edt, out=sl)

    # Normalise to [0, 1] by global max for cross-sensor consistency.
    max_val = weight.max()
    if max_val > 0:
        weight /= max_val

    return np.clip(weight * 255, 0, 255).astype(np.uint8)
