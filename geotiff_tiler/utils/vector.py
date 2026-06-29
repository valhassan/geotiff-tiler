"""
Geographic-space vector utilities for GeoJSON patch export.

Clips vector features to patch window extents and serializes them as
georeferenced GeoJSON — same CRS as the source image and raster label —
so patches overlay correctly in QGIS and any GIS tool without coordinate
transforms.  Pixel-space conversion (for COCO / YOLO export) is an
offline post-processing step done from the stored GeoJSON + patch
geotransform.
"""

import json
import logging

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import box

logger = logging.getLogger(__name__)

try:
    import orjson

    def _dumps(obj: dict) -> str:
        return orjson.dumps(obj).decode()

except ImportError:
    def _dumps(obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":"))


def clip_gdf_to_window(
    label_gdf: gpd.GeoDataFrame,
    window: Window,
    src_transform: Affine,
) -> gpd.GeoDataFrame:
    """Return rows of *label_gdf* whose geometries intersect the given raster window.

    Uses the spatial index (STRtree) for O(log n) candidate selection rather than
    iterating all features per patch.

    Args:
        label_gdf: Full-image GeoDataFrame in geographic CRS.
        window: Rasterio Window (col_off, row_off, width, height).
        src_transform: Image-level affine transform (not window transform).

    Returns:
        Subset GeoDataFrame intersecting the window extent (geographic CRS preserved).
    """
    if label_gdf is None or label_gdf.empty:
        return label_gdf

    win_transform = src_transform * Affine.translation(window.col_off, window.row_off)
    bounds = rasterio.transform.array_bounds(window.height, window.width, win_transform)
    patch_box = box(*bounds)

    hits = label_gdf.sindex.query(patch_box, predicate="intersects")
    return label_gdf.iloc[hits].reset_index(drop=True)


def gdf_to_geojson(
    label_gdf: gpd.GeoDataFrame,
    window: Window,
    src_transform: Affine,
    coord_precision: float = 0.01,
    drop_cols: tuple[str, ...] = ("geometry", "extent_geometry", "burn_val"),
) -> str:
    """Serialize a spatially-filtered GeoDataFrame as a georeferenced GeoJSON string.

    Features are clipped to the window extent and tagged with ``is_truncated``
    (True when the original polygon extended beyond the patch boundary).
    Coordinates are snapped to *coord_precision* CRS units (default 0.01 m for
    projected CRS) to reduce file size without meaningful loss of precision.

    The output CRS matches the source image — the GeoJSON overlays correctly on
    the co-located GeoTIFF patch in QGIS or any GIS tool.

    Args:
        label_gdf: GeoDataFrame already spatially filtered to the patch window
            (output of :func:`clip_gdf_to_window`).
        window: Rasterio Window used to derive the patch geographic extent.
        src_transform: Image-level affine transform.
        coord_precision: Snap-to-grid size in CRS units.
        drop_cols: Columns to exclude from feature properties.

    Returns:
        Compact GeoJSON FeatureCollection string in the source image CRS.
    """
    if label_gdf is None or label_gdf.empty:
        return _dumps({"type": "FeatureCollection", "features": []})

    # Geographic bounding box of this patch window
    win_transform = src_transform * Affine.translation(window.col_off, window.row_off)
    bounds = rasterio.transform.array_bounds(window.height, window.width, win_transform)
    patch_box_geo = box(*bounds)

    prop_cols = [c for c in label_gdf.columns if c not in drop_cols]

    features = []
    for idx in range(len(label_gdf)):
        geom = label_gdf.geometry.iloc[idx]

        if not geom.is_valid:
            geom = geom.make_valid()

        # Clip geometry to patch extent (handles straddling features)
        geom = geom.intersection(patch_box_geo)
        if geom.is_empty:
            continue

        # Truncation: original polygon extended beyond the patch boundary
        is_truncated = not patch_box_geo.contains(geom)

        # Reduce coordinate precision — suppresses float64 bloat
        geom = shapely.set_precision(geom, grid_size=coord_precision)

        props: dict = {"is_truncated": bool(is_truncated)}
        row = label_gdf.iloc[idx]
        for col in prop_cols:
            val = row[col]
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            props[col] = val

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": geom.__geo_interface__,
            }
        )

    fc: dict = {"type": "FeatureCollection", "features": features}
    if label_gdf.crs is not None:
        epsg = label_gdf.crs.to_epsg()
        if epsg:
            fc["crs"] = {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
            }
    return _dumps(fc)
