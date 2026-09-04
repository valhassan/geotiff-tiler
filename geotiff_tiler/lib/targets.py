"""Building and road supervision targets from vector polygons."""

from __future__ import annotations

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
from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)

_SDF_CLIP_M = 32.0
_ROAD_REF_M = 20.0


def _pixel_coords(ring, transform: Affine):
    sx, sy = transform.a, transform.e
    ox, oy = transform.c, transform.f
    coords = np.asarray(ring.coords)
    px = (coords[:, 0] - ox) / sx
    py = (coords[:, 1] - oy) / sy
    return px, py


def _rings(geom: Polygon):
    yield geom.exterior
    yield from geom.interiors


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
    sigma: float = 3.0,
    max_dist_meters: float = 10.0,
    vertex_sigma: float = 1.5,
    max_gsd_for_targets: float = 1.0,
) -> dict[str, str]:
    """
    Compute all four building supervision targets from polygon geometry.

    Args:
        building_gdf:         GeoDataFrame filtered to building polygons only.
        image_path:           Path to the source image (for transform/shape).
        tmp_dir:              Temp directory for output tifs.
        label_name:           Stem used for output filenames.
        sigma:                EDT decay sigma for dual-distance weight map (metres).
        max_dist_meters:      Maximum inter-instance distance to consider (metres).
        vertex_sigma:         Gaussian sigma for vertex heatmap (pixels).
        max_gsd_for_targets:  GSD threshold in metres above which targets are
                              skipped. Default matches the erosion threshold.

    Returns:
        Dict mapping target name → tif path. Keys:
            'edt'       dual-distance boundary weight map  uint8
            'boundary'  vector boundary map                uint8
            'vertices'  vertex heatmap                     uint8
            'sdf'       signed distance field              float32
    Returns {} when GSD is too coarse.
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs
        h, w = src.height, src.width

    pixel_size = abs(transform.a)
    if pixel_size > max_gsd_for_targets:
        logger.info(
            "[building_targets] skipping — GSD %.2fm > threshold %.2fm",
            pixel_size,
            max_gsd_for_targets,
        )
        return {}

    max_dist_px = max_dist_meters / pixel_size

    valid_geoms = _explode_to_polygons(
        building_gdf[
            ~building_gdf.geometry.is_empty & building_gdf.geometry.notnull()
        ].geometry.tolist()
    )

    t = time.time()
    edt_map = _compute_dual_distance_edt(
        valid_geoms, h, w, transform, max_dist_px, sigma
    )
    logger.info("EDT:      %.1fs", time.time() - t)
    paths = {}
    _write_target(
        paths, tmp_dir, label_name, "edt", edt_map, "uint8", h, w, crs, transform
    )
    del edt_map

    t = time.time()
    boundary_map = _compute_vector_boundary(valid_geoms, h, w, transform)
    logger.info("Boundary: %.1fs", time.time() - t)
    _write_target(
        paths,
        tmp_dir,
        label_name,
        "boundary",
        np.clip(boundary_map * 255, 0, 255).astype(np.uint8),
        "uint8",
        h,
        w,
        crs,
        transform,
    )
    del boundary_map

    t = time.time()
    vertex_map = _compute_vertex_heatmap(valid_geoms, h, w, transform, vertex_sigma)
    logger.info("Vertices: %.1fs", time.time() - t)
    _write_target(
        paths,
        tmp_dir,
        label_name,
        "vertices",
        np.clip(vertex_map * 255, 0, 255).astype(np.uint8),
        "uint8",
        h,
        w,
        crs,
        transform,
    )
    del vertex_map

    t = time.time()
    sdf_map = _compute_sdf(valid_geoms, h, w, transform)
    logger.info("SDF:      %.1fs", time.time() - t)
    _write_target(
        paths, tmp_dir, label_name, "sdf", sdf_map, "float32", h, w, crs, transform
    )
    return paths


def _write_target(
    paths: dict[str, str],
    tmp_dir: str,
    label_name: str,
    key: str,
    arr: np.ndarray,
    dtype: str,
    h: int,
    w: int,
    crs,
    transform: Affine,
) -> None:
    out_path = Path(tmp_dir) / f"{label_name}_buildings_{key}.tif"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(arr, 1)
    paths[key] = str(out_path)
    logger.info("[building_targets] wrote %s → %s", key, out_path)


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
        r0 = max(0, int((maxy - transform.f) / transform.e) - int(max_dist_px) - 1)
        c1 = min(w, int((maxx - transform.c) / transform.a) + int(max_dist_px) + 2)
        r1 = min(h, int((miny - transform.f) / transform.e) + int(max_dist_px) + 2)

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


def _compute_vector_boundary(
    geoms,
    h: int,
    w: int,
    transform: Affine,
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
        all_xs, all_ys = [], []
        for ring in _rings(geom):
            px, py = _pixel_coords(ring, transform)
            for i in range(len(px) - 1):
                length = np.hypot(px[i + 1] - px[i], py[i + 1] - py[i])
                n = max(int(length * 2), 1)
                ts = np.linspace(0, 1, n)
                all_xs.append(px[i] + ts * (px[i + 1] - px[i]))
                all_ys.append(py[i] + ts * (py[i + 1] - py[i]))

        if not all_xs:
            continue

        xs = np.concatenate(all_xs)  # (M,)
        ys = np.concatenate(all_ys)  # (M,)

        # Clip sample centres to valid range
        cjs = np.clip(xs.astype(int), r, w - r - 1)
        cis = np.clip(ys.astype(int), r, h - r - 1)

        # Neighbourhood offsets
        off = np.arange(-r, r + 1)
        di, dj = np.meshgrid(off, off, indexing="ij")  # (5,5)
        di = di.ravel()  # (25,)
        dj = dj.ravel()  # (25,)

        # Vectorized: (M, 25) index arrays
        ni = cis[:, None] + di[None, :]  # (M, 25)
        nj = cjs[:, None] + dj[None, :]  # (M, 25)
        dist2 = (nj - xs[:, None]) ** 2 + (ni - ys[:, None]) ** 2  # (M, 25)
        weights = np.exp(-dist2 / (2 * sigma_px**2))  # (M, 25)

        np.add.at(boundary, (ni.ravel(), nj.ravel()), weights.ravel())

    return np.clip(boundary, 0, 1)


def _compute_vertex_heatmap(
    geoms,
    h: int,
    w: int,
    transform: Affine,
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
        for ring in _rings(geom):
            px, py = _pixel_coords(ring, transform)
            all_vx.append(px[:-1])
            all_vy.append(py[:-1])

    if not all_vx:
        return heatmap

    vx = np.concatenate(all_vx)  # (V,)
    vy = np.concatenate(all_vy)  # (V,)

    # Clip centres
    cjs = np.clip(vx.astype(int), r, w - r - 1)
    cis = np.clip(vy.astype(int), r, h - r - 1)

    off = np.arange(-r, r + 1)
    di, dj = np.meshgrid(off, off, indexing="ij")
    di, dj = di.ravel(), dj.ravel()  # (K,)

    ni = cis[:, None] + di[None, :]  # (V, K)
    nj = cjs[:, None] + dj[None, :]  # (V, K)
    dist2 = (nj - vx[:, None]) ** 2 + (ni - vy[:, None]) ** 2
    weights = np.exp(-dist2 / (2 * sigma**2))

    np.add.at(heatmap, (ni.ravel(), nj.ravel()), weights.ravel())

    return np.clip(heatmap, 0, 1)


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
    Scaled by metres / _SDF_CLIP_M and clipped to [-1, 1].
    Stored as float32.
    """
    if not geoms:
        return np.zeros((h, w), dtype=np.float32)

    all_mask = rasterize(
        [(g, 1) for g in geoms],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    sdf = distance_transform_edt(all_mask).astype(np.float32)
    exterior = distance_transform_edt(~all_mask).astype(np.float32)
    np.copyto(sdf, -exterior, where=~all_mask)
    del exterior, all_mask
    sdf *= abs(transform.a)
    np.clip(sdf / _SDF_CLIP_M, -1.0, 1.0, out=sdf)
    return sdf


def compute_road_targets(
    road_gdf: gpd.GeoDataFrame,
    image_path: str,
    tmp_dir: str,
    label_name: str,
    max_gsd_for_targets: float = 1.0,
) -> dict[str, str]:
    """
    Compute road supervision targets from polygon geometry.

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
        tiled=True,
        blockxsize=256,
        blockysize=256,
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

    Normalised by _ROAD_REF_M metres so scale is consistent across scenes.

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
        r0 = max(0, int((maxy - transform.f) / transform.e) - 1)
        c1 = min(w, int((maxx - transform.c) / transform.a) + 2)
        r1 = min(h, int((miny - transform.f) / transform.e) + 2)

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
        local_edt *= abs(transform.a)

        sl = weight[r0:r1, c0:c1]
        np.maximum(sl, local_edt, out=sl)

    np.clip(weight / _ROAD_REF_M, 0, 1, out=weight)
    return (weight * 255).astype(np.uint8)
