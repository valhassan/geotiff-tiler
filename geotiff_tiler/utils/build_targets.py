"""
Precomputed building supervision targets derived from vector polygon geometry.
All targets are computed at full image resolution and sliced at patch time.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, MultiPolygon

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
        crs = src.crs
        h, w = src.height, src.width

    pixel_size = abs(transform.a)
    max_dist_px = max_dist_meters / pixel_size

    valid_geoms = _explode_to_polygons(building_gdf[
        ~building_gdf.geometry.is_empty & building_gdf.geometry.notnull()
    ].geometry.tolist())

    # Compute all four targets in a single pass over geometries
    t = time.time()
    edt_map = _compute_dual_distance_edt(
        valid_geoms, h, w, transform, max_dist_px, sigma
    )
    logger.info(f"EDT:      {time.time() - t:.1f}s")
    t = time.time()
    boundary_map = _compute_vector_boundary(valid_geoms, h, w, transform)
    logger.info(f"Boundary: {time.time() - t:.1f}s")
    t = time.time()
    vertex_map = _compute_vertex_heatmap(valid_geoms, h, w, transform, vertex_sigma)
    logger.info(f"Vertices: {time.time() - t:.1f}s")
    t = time.time()
    sdf_map = _compute_sdf(valid_geoms, h, w, transform)
    logger.info(f"SDF:      {time.time() - t:.1f}s")

    # Write to tif files
    paths = {}
    specs = [
        ("edt", edt_map, "uint8", np.uint8),
        ("boundary", np.clip(boundary_map * 255, 0, 255), "uint8", np.uint8),
        ("vertices", np.clip(vertex_map * 255, 0, 255), "uint8", np.uint8),
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
            crs=crs,
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
    geoms, h: int, w: int, transform: Affine,
) -> np.ndarray:
    """Compute the vector boundary map.
    Args:
        geoms: List of geometries.
        h: Height of the image.
        w: Width of the image.
        transform: Transform of the image.
    Returns:
        np.ndarray: Vector boundary map.
    """
    boundary = np.zeros((h, w), dtype=np.float32)
    sigma_px = 0.8
    r = 2

    for geom in geoms:
        coords = np.array(geom.exterior.coords)
        px = (coords[:, 0] - transform.c) / transform.a
        py = (coords[:, 1] - transform.f) / transform.e

        # Collect all sample points across all edges at once
        all_xs, all_ys = [], []
        for i in range(len(px) - 1):
            length = np.hypot(px[i+1] - px[i], py[i+1] - py[i])
            n = max(int(length * 2), 1)
            ts = np.linspace(0, 1, n)
            all_xs.append(px[i] + ts * (px[i+1] - px[i]))
            all_ys.append(py[i] + ts * (py[i+1] - py[i]))

        if not all_xs:
            continue

        xs = np.concatenate(all_xs)   # (M,)
        ys = np.concatenate(all_ys)   # (M,)

        # Clip sample centres to valid range
        cjs = np.clip(xs.astype(int), r, w - r - 1)
        cis = np.clip(ys.astype(int), r, h - r - 1)

        # Neighbourhood offsets
        off = np.arange(-r, r + 1)
        di, dj = np.meshgrid(off, off, indexing='ij')  # (5,5)
        di = di.ravel()   # (25,)
        dj = dj.ravel()   # (25,)

        # Vectorized: (M, 25) index arrays
        ni = cis[:, None] + di[None, :]   # (M, 25)
        nj = cjs[:, None] + dj[None, :]   # (M, 25)
        dist2 = (nj - xs[:, None]) ** 2 + (ni - ys[:, None]) ** 2  # (M, 25)
        weights = np.exp(-dist2 / (2 * sigma_px ** 2))              # (M, 25)

        np.add.at(boundary, (ni.ravel(), nj.ravel()), weights.ravel())

    return np.clip(boundary, 0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Target 3: Vertex heatmap
# ─────────────────────────────────────────────────────────────────────────────


def _compute_vertex_heatmap(
    geoms, h: int, w: int, transform: Affine,
    sigma: float = 1.5,
) -> np.ndarray:
    """Compute the vertex heatmap.
    Args:
        geoms: List of geometries.
        h: Height of the image.
        w: Width of the image.
        transform: Transform of the image.
        sigma: Sigma of the Gaussian.
    Returns:
        np.ndarray: Vertex heatmap.
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    r = int(np.ceil(3 * sigma))

    # Collect ALL vertices across all geometries at once
    all_vx, all_vy = [], []
    for geom in geoms:
        coords = np.array(geom.exterior.coords[:-1])
        all_vx.append((coords[:, 0] - transform.c) / transform.a)
        all_vy.append((coords[:, 1] - transform.f) / transform.e)

    if not all_vx:
        return heatmap

    vx = np.concatenate(all_vx)   # (V,)
    vy = np.concatenate(all_vy)   # (V,)

    # Clip centres
    cjs = np.clip(vx.astype(int), r, w - r - 1)
    cis = np.clip(vy.astype(int), r, h - r - 1)

    off = np.arange(-r, r + 1)
    di, dj = np.meshgrid(off, off, indexing='ij')
    di, dj = di.ravel(), dj.ravel()          # (K,)

    ni = cis[:, None] + di[None, :]          # (V, K)
    nj = cjs[:, None] + dj[None, :]          # (V, K)
    dist2 = (nj - vx[:, None]) ** 2 + (ni - vy[:, None]) ** 2
    weights = np.exp(-dist2 / (2 * sigma ** 2))

    np.add.at(heatmap, (ni.ravel(), nj.ravel()), weights.ravel())

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
