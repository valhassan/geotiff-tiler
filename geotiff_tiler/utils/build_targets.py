"""
Precomputed building supervision targets derived from vector polygon geometry.
All targets are computed at full image resolution and sliced at patch time.
"""

import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def _pixel_coords(geom, transform: Affine):
    """Convert a geometry from CRS space to pixel float coordinates."""
    sx, sy = transform.a, transform.e  # pixel width, pixel height (negative)
    ox, oy = transform.c, transform.f  # origin x, origin y
    coords = np.array(geom.exterior.coords)  # (N, 2) float
    px = (coords[:, 0] - ox) / sx
    py = (coords[:, 1] - oy) / sy
    return px, py  # float pixel coords


def _rasterize_geom(
    geom, shape: Tuple[int, int], local_transform: Affine
) -> np.ndarray:
    """Rasterize a single geometry into a local boolean mask."""
    return rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=local_transform,
        fill=0,
        dtype=np.uint8,
    )


def compute_building_targets(
    building_gdf: gpd.GeoDataFrame,
    image_path: str,
    tmp_dir: str,
    label_name: str,
    building_class_val: int = 4,
    sigma: float = 3.0,
    max_dist_meters: float = 10.0,
    vertex_sigma: float = 1.5,
) -> dict[str, str]:
    """
    Compute all four building supervision targets from polygon geometry.

    Args:
        building_gdf:       GeoDataFrame filtered to building polygons only.
        image_path:         Path to the source image (for transform/shape).
        tmp_dir:            Temp directory for output tifs.
        label_name:         Stem used for output filenames.
        building_class_val: Integer class id for buildings in the label raster.
        sigma:              EDT decay sigma for dual-distance weight map (metres).
        max_dist_meters:    Maximum inter-instance distance to consider (metres).
        vertex_sigma:       Gaussian sigma for vertex heatmap (pixels).

    Returns:
        Dict mapping target name → tif path. Keys:
            'edt'       dual-distance boundary weight map  uint8
            'boundary'  vector boundary map                float32
            'vertices'  vertex heatmap                     float32
            'sdf'       signed distance field              float32
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
        h, w = src.height, src.width

    pixel_size = abs(transform.a)
    max_dist_px = max_dist_meters / pixel_size

    valid_geoms = building_gdf[
        ~building_gdf.geometry.is_empty & building_gdf.geometry.notnull()
    ].geometry.tolist()

    # Compute all four targets in a single pass over geometries
    edt_map = _compute_dual_distance_edt(
        valid_geoms, h, w, transform, max_dist_px, sigma
    )
    boundary_map = _compute_vector_boundary(valid_geoms, h, w, transform)
    vertex_map = _compute_vertex_heatmap(valid_geoms, h, w, transform, vertex_sigma)
    sdf_map = _compute_sdf(valid_geoms, h, w, transform)

    # Write to tif files
    paths = {}
    specs = [
        ("edt", edt_map, "uint8", np.uint8),
        ("boundary", boundary_map, "float32", np.float32),
        ("vertices", vertex_map, "float32", np.float32),
        ("sdf", sdf_map, "float32", np.float32),
    ]

    for key, arr, dtype_str, dtype_np in specs:
        out_path = Path(tmp_dir) / f"{label_name}_buildings_{key}.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=dtype_str,
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(arr.astype(dtype_np), 1)
        paths[key] = str(out_path)
        logger.info(f"[building_targets] wrote {key} → {out_path}")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Target 1: Dual-distance EDT weight map
# ─────────────────────────────────────────────────────────────────────────────


def _compute_dual_distance_edt(
    geoms,
    h: int,
    w: int,
    transform: Affine,
    max_dist_px: float,
    sigma: float,
) -> np.ndarray:
    """
    Per-pixel weight = exp(-(d1+d2)/sigma) where d1, d2 are distances
    to the nearest and second-nearest building boundaries.
    Localised EDT per polygon for speed. Stored as uint8 [0-255].
    """
    d1 = np.full((h, w), max_dist_px, dtype=np.float32)
    d2 = np.full((h, w), max_dist_px, dtype=np.float32)

    for geom in geoms:
        minx, miny, maxx, maxy = geom.bounds
        # Pad by max_dist_px
        c0 = max(0, int((minx - transform.c) / transform.a) - int(max_dist_px) - 1)
        r0 = max(0, int((miny - transform.f) / transform.e) - int(max_dist_px) - 1)
        c1 = min(w, int((maxx - transform.c) / transform.a) + int(max_dist_px) + 2)
        r1 = min(h, int((maxy - transform.f) / transform.e) + int(max_dist_px) + 2)

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

        local_dist = distance_transform_edt(1 - mask).astype(np.float32)

        sl_d1 = d1[r0:r1, c0:c1]
        sl_d2 = d2[r0:r1, c0:c1]

        closer_than_d1 = local_dist < sl_d1
        between = (~closer_than_d1) & (local_dist < sl_d2)

        sl_d2[closer_than_d1] = sl_d1[closer_than_d1]
        sl_d1[closer_than_d1] = local_dist[closer_than_d1]
        sl_d2[between] = local_dist[between]

    both_valid = (d1 < max_dist_px) & (d2 < max_dist_px)
    weight = np.zeros((h, w), dtype=np.float32)
    weight[both_valid] = np.exp(-(d1[both_valid] + d2[both_valid]) / sigma)

    return np.clip(weight * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Target 2: Vector boundary map
# ─────────────────────────────────────────────────────────────────────────────


def _compute_vector_boundary(
    geoms,
    h: int,
    w: int,
    transform: Affine,
) -> np.ndarray:
    """
    Soft boundary map from polygon edge segments at sub-pixel precision.
    Each edge pixel receives a Gaussian weighted by distance to the exact
    polygon edge line — not a morphological approximation.
    Stored as float32 [0, 1].
    """
    boundary = np.zeros((h, w), dtype=np.float32)
    sigma_px = 0.8  # sub-pixel spread

    for geom in geoms:
        coords = np.array(geom.exterior.coords)  # (N, 2) CRS coords
        # Convert to float pixel coords
        px = (coords[:, 0] - transform.c) / transform.a
        py = (coords[:, 1] - transform.f) / transform.e

        for i in range(len(px) - 1):
            _splat_segment(boundary, px[i], py[i], px[i + 1], py[i + 1], sigma_px, h, w)

    return np.clip(boundary, 0, 1)


def _splat_segment(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sigma: float,
    h: int,
    w: int,
) -> None:
    """
    Rasterize a line segment as a Gaussian-weighted soft line.
    Samples points along the segment at 0.5px intervals and splatted
    each sample onto the canvas in-place.
    """
    length = np.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return

    n_steps = max(int(length * 2), 1)  # 0.5px spacing
    ts = np.linspace(0, 1, n_steps)
    xs = x0 + ts * (x1 - x0)
    ys = y0 + ts * (y1 - y0)

    # Integer pixel neighbourhood for each sample
    for sx, sy in zip(xs, ys):
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni = int(sy) + di
                nj = int(sx) + dj
                if 0 <= ni < h and 0 <= nj < w:
                    dist2 = (nj - sx) ** 2 + (ni - sy) ** 2
                    canvas[ni, nj] += np.exp(-dist2 / (2 * sigma**2))


# ─────────────────────────────────────────────────────────────────────────────
# Target 3: Vertex heatmap
# ─────────────────────────────────────────────────────────────────────────────


def _compute_vertex_heatmap(
    geoms,
    h: int,
    w: int,
    transform: Affine,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Gaussian blob at each polygon vertex at exact sub-pixel float coordinates.
    Encodes the rectilinear corner prior for buildings.
    Stored as float32 [0, 1].
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    r = int(np.ceil(3 * sigma))  # splat radius

    for geom in geoms:
        coords = np.array(geom.exterior.coords[:-1])  # drop closing duplicate
        px = (coords[:, 0] - transform.c) / transform.a
        py = (coords[:, 1] - transform.f) / transform.e

        for vx, vy in zip(px, py):
            ci, cj = int(vy), int(vx)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        dist2 = (nj - vx) ** 2 + (ni - vy) ** 2
                        heatmap[ni, nj] += np.exp(-dist2 / (2 * sigma**2))

    return np.clip(heatmap, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Target 4: Signed Distance Field
# ─────────────────────────────────────────────────────────────────────────────


def _compute_sdf(
    geoms,
    h: int,
    w: int,
    transform: Affine,
) -> np.ndarray:
    """
    Per-pixel signed distance to the nearest building polygon boundary.
    Positive  = inside polygon  (distance to boundary from interior)
    Negative  = outside polygon (distance to nearest polygon)
    Zero      = exactly on boundary.
    Normalised per-image to [-1, 1].
    Stored as float32.
    """
    if not geoms:
        return np.zeros((h, w), dtype=np.float32)

    # Rasterize all building polygons together for the binary mask
    all_mask = rasterize(
        [(g, 1) for g in geoms],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    interior_dist = distance_transform_edt(all_mask).astype(np.float32)
    exterior_dist = distance_transform_edt(~all_mask).astype(np.float32)

    sdf = np.where(all_mask, interior_dist, -exterior_dist)

    # Normalize to [-1, 1] for training stability
    max_abs = max(np.abs(sdf).max(), 1.0)
    sdf /= max_abs

    return sdf.astype(np.float32)
