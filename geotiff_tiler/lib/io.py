import functools
import gc
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import pystac
import rasterio
import rasterio.features
from rasterio.enums import MaskFlags
from rasterio.warp import transform_bounds
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from shapely.geometry import box, shape

from .geo import (
    SingleBandItemEO,
    check_alignment,
    check_image_validity,
    check_label_type,
    check_label_validity,
    is_image_georeferenced,
    is_label_georeferenced,
    select_bands,
    stack_bands,
    with_connection_retry,
)
from .targets import compute_building_targets, compute_road_targets

logger = logging.getLogger(__name__)

IGNORE_INDEX = 255


def mask_is_declared(src: rasterio.DatasetReader) -> bool:
    return any(
        MaskFlags.all_valid not in flags for flags in src.mask_flag_enums
    )


def nodata_spec(src: rasterio.DatasetReader) -> tuple[float, str]:
    if mask_is_declared(src):
        nd = src.nodata
        return (float(nd) if nd is not None else 0.0), "declared"
    return 0.0, "fallback_zero"


def window_valid(
    src: rasterio.DatasetReader, window: Window | None = None
) -> np.ndarray:
    if mask_is_declared(src):
        m = src.read_masks(window=window)
        if m.ndim == 3:
            return np.any(m > 0, axis=0)
        return m > 0
    data = src.read(window=window)
    return np.any(np.isfinite(data) & (data != 0), axis=0)


def image_nodata_meta(image_path: str) -> tuple[str, float]:
    with rasterio.open(image_path) as src:
        _, source = nodata_spec(src)
        n_ok = n = 0
        for _, win in src.block_windows(1):
            valid = window_valid(src, win)
            n_ok += int(valid.sum())
            n += valid.size
    return source, (n_ok / n if n else 0.0)


def label_src_nodata(label_path: str) -> int | float | str:
    with rasterio.open(label_path) as src:
        nd = src.nodata
    if nd is None or nd == 0:
        return "None"
    return nd


def require_class_ids(class_ids: dict | None) -> None:
    if not class_ids:
        return
    bad = {k: v for k, v in class_ids.items() if int(v) < 0 or int(v) >= IGNORE_INDEX}
    if bad:
        raise ValueError(f"class ids must be in 0–254, got {bad}")


def gdal_translate_copy(src, dst):
    result = subprocess.run(
        ["gdal_translate", "-of", "GTiff", src, dst], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"GDAL failed: {result.stderr}")


def log_stage(stage_name=None, log_memory=False, force_gc=False):
    """Decorator to log time and memory usage of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = stage_name or func.__name__
            if force_gc:
                gc.collect()
            start_time = time.time()
            process = psutil.Process()
            mem_before = process.memory_info().rss

            logger.info(f"[{name}] started...")

            result = func(*args, **kwargs)

            duration = time.time() - start_time
            mem_after = process.memory_info().rss

            logger.info(f"[{name}] completed in {duration:.2f}s")

            if log_memory:
                delta = (mem_after - mem_before) / 1024**2
                logger.info(
                    f"[{name}] memory change: {delta:.2f} MB (now: {mem_after / 1024**2:.2f} MB)"
                )

            return result

        return wrapper

    return decorator


def resolve_attr_field(
    columns: Sequence[str], attr_field: str | Sequence[str] | None
) -> str | None:
    """First requested field that exists in *columns*."""
    if attr_field is None:
        return None
    fields = [attr_field] if isinstance(attr_field, str) else list(attr_field)
    for name in fields:
        if name in columns:
            return name
    return None


def _repair_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """make_valid, then drop leftovers that are still invalid or empty."""
    if gdf.empty or gdf.geometry.is_valid.all():
        return gdf
    logger.info("Found invalid geometries, fixing with make_valid()")
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    if not gdf.geometry.is_valid.all():
        invalid_count = int((~gdf.geometry.is_valid).sum())
        logger.warning(
            "Filtering out %s geometries that remain invalid after make_valid()",
            invalid_count,
        )
        gdf = gdf[gdf.geometry.is_valid].copy()
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        empty_count = int((gdf.geometry.isna() | gdf.geometry.is_empty).sum())
        logger.warning("Filtering out %s empty/null geometries", empty_count)
        gdf = gdf[~(gdf.geometry.isna() | gdf.geometry.is_empty)].copy()
    return gdf


def load_vector_mask(
    mask_path: str, skip_layer: str | None = "extent"
) -> gpd.GeoDataFrame:
    """Load a vector mask. GeoPackage uses the non-extent layer; other formats via GeoPandas."""
    if Path(mask_path).suffix.lower() != ".gpkg":
        return _repair_geoms(gpd.read_file(mask_path))
    layers = fiona.listlayers(mask_path)
    extent_layer = next((layer for layer in layers if "extent" in layer.lower()), None)
    main_layer = next(
        (layer for layer in layers if skip_layer not in layer.lower()), None
    )
    if main_layer is None:
        raise ValueError(f"No suitable layer found in {mask_path}")
    result = _repair_geoms(gpd.read_file(mask_path, layer=main_layer))

    if extent_layer:
        extent_gdf = gpd.read_file(mask_path, layer=extent_layer)
        if not extent_gdf.empty:
            if extent_gdf.crs != result.crs:
                logger.warning(
                    f"Extent layer CRS ({extent_gdf.crs}) doesn't match main layer CRS ({result.crs}). Reprojecting extent layer."
                )
                extent_gdf = extent_gdf.to_crs(result.crs)
            extent_geom = extent_gdf.geometry.iloc[0]
            if not extent_geom.is_valid:
                logger.info("Found invalid extent geometry, fixing with make_valid()")
                extent_geom = extent_geom.make_valid()
            if extent_geom.is_valid and not extent_geom.is_empty:
                result.attrs["extent_geometry"] = extent_geom
    return result


def save_vector_mask(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    extent_geometry: gpd.GeoSeries | None = None,
    main_layer: str = "labels",
    extent_layer: str = "extent",
):
    """Saves a vector mask to a path."""
    gdf.to_file(output_path, layer=main_layer, driver="GPKG")
    if extent_geometry is not None:
        extent_gdf = gpd.GeoDataFrame(geometry=[extent_geometry], crs=gdf.crs)
        extent_gdf.to_file(output_path, layer=extent_layer, driver="GPKG")
        del extent_gdf
    del gdf


def _looks_like_stac(image_path: str) -> bool:
    s = str(image_path).lower()
    return s.startswith(("http://", "https://")) or s.endswith((".json", ".jsonl"))


def _local_image(
    image_path: str,
    band_indices: Sequence | None,
    tmp_dir: str | Path | None,
) -> str:
    if not band_indices:
        return image_path
    if tmp_dir is None:
        raise ValueError("band_indices requires tmp_dir")
    out = Path(tmp_dir) / f"{Path(image_path).stem}_bands.vrt"
    return select_bands(image_path, band_indices, out)


@with_connection_retry
@log_stage(stage_name="validate_image")
def validate_image(
    image_path: str,
    bands_requested: Sequence = ["red", "green", "blue"],
    band_indices: Sequence | None = None,
    tmp_dir: str | Path | None = None,
):
    """Validates an image from a path or stac item"""
    local = Path(image_path).exists()
    if local and not _looks_like_stac(image_path):
        return _local_image(image_path, band_indices, tmp_dir)

    try:
        stac_item = pystac.Item.from_file(image_path)
        item = SingleBandItemEO(item=stac_item, bands_requested=bands_requested)
        stac_bands = [value["meta"].href for value in item.bands_requested.values()]
        return stack_bands(stac_bands)
    except Exception:
        if local:
            return _local_image(image_path, band_indices, tmp_dir)
        raise FileNotFoundError(f"File not found: {image_path}") from None


def validate_mask(mask_path: str):
    """Validates a mask from a path"""
    if Path(mask_path).exists():
        label_type = check_label_type(mask_path)
        return mask_path, label_type
    else:
        raise FileNotFoundError(f"File not found: {mask_path}")


@log_stage(stage_name="validate_pair")
def validate_pair(image_path, label_path, label_type):
    """Validates an image-label pair based on georeferencing and data integrity."""

    with rasterio.open(image_path) as src_image:
        logger.info("Validating image in pair")
        image_valid, image_msg = check_image_validity(src_image)
        if not image_valid:
            return {
                "valid": False,
                "special_case": False,
                "reason": f"Invalid image: {image_msg}",
            }

        if label_type == "vector":
            logger.info("Validating vector label in pair")
            label_gdf = load_vector_mask(label_path)
            label_valid, label_msg = check_label_validity(label_gdf)
            if not label_valid:
                return {
                    "valid": False,
                    "special_case": False,
                    "reason": f"Invalid label: {label_msg}",
                }

            if not is_image_georeferenced(src_image) or not is_label_georeferenced(
                label_gdf
            ):
                return {
                    "valid": False,
                    "special_case": False,
                    "reason": "Invalid georeferencing for vector label or image",
                }

        elif label_type == "raster":
            with rasterio.open(label_path) as src_label:
                label_valid, label_msg = check_label_validity(src_label)
                if not label_valid:
                    return {
                        "valid": False,
                        "special_case": False,
                        "reason": f"Invalid label: {label_msg}",
                    }
                if not is_image_georeferenced(src_image) or not is_label_georeferenced(
                    src_label
                ):
                    if check_alignment(src_image, src_label):
                        return {
                            "valid": True,
                            "special_case": True,
                            "reason": "Non-georeferenced but aligned raster pair",
                        }
                    return {
                        "valid": False,
                        "special_case": False,
                        "reason": "Invalid georeferencing or alignment for raster label or image",
                    }

    return {"valid": True, "reason": "Valid pair", "special_case": False}


def _pair_geoms(image_path: str, label_path: str, label_type: str):
    """Image and label extents as shapely geometries in the image CRS."""
    with rasterio.open(image_path) as image:
        image_bounds = box(*image.bounds)
        image_crs = image.crs
    if label_type == "raster":
        with rasterio.open(label_path) as label:
            if image_crs and label.crs and image_crs != label.crs:
                return image_bounds, box(
                    *transform_bounds(label.crs, image_crs, *label.bounds)
                )
            return image_bounds, box(*label.bounds)
    label = load_vector_mask(label_path)
    geom = label.attrs.get("extent_geometry") if hasattr(label, "attrs") else None
    if geom is not None:
        if image_crs and label.crs and label.crs != image_crs:
            geom = gpd.GeoSeries([geom], crs=label.crs).to_crs(image_crs).iloc[0]
        return image_bounds, geom
    if image_crs and label.crs and label.crs != image_crs:
        label = label.to_crs(image_crs)
    return image_bounds, box(*label.total_bounds)


@log_stage(stage_name="calculate_overlap", log_memory=True)
def calculate_overlap(
    image_path: str, label_path: str, label_type: str
) -> Tuple[float, str]:
    """Calculate the overlap between image and label data."""
    image_bounds, label_bounds = _pair_geoms(image_path, label_path, label_type)
    intersection_area = image_bounds.intersection(label_bounds).area
    union_area = image_bounds.union(label_bounds).area
    if union_area == 0:
        return 0.0, "No valid area found"
    overlap_percentage = (intersection_area / union_area) * 100
    if overlap_percentage == 0:
        return 0.0, "No overlap between image and label"
    return overlap_percentage, f"Overlap percentage: {overlap_percentage:.2f}%"


def _pixel_aligned_extent(
    src: rasterio.DatasetReader,
    geometry,
) -> Tuple[float, float, float, float, float, float]:
    if src.transform.b != 0 or src.transform.d != 0:
        raise ValueError("Rotated transforms are not supported")
    if hasattr(geometry, "geom_type"):
        bounds = geometry.bounds
    else:
        geoms = list(geometry)
        bounds = (
            min(g.bounds[0] for g in geoms),
            min(g.bounds[1] for g in geoms),
            max(g.bounds[2] for g in geoms),
            max(g.bounds[3] for g in geoms),
        )
    win = from_bounds(*bounds, transform=src.transform)
    col0 = max(int(math.floor(win.col_off + 1e-6)), 0)
    row0 = max(int(math.floor(win.row_off + 1e-6)), 0)
    col1 = min(int(math.ceil(win.col_off + win.width - 1e-6)), src.width)
    row1 = min(int(math.ceil(win.row_off + win.height - 1e-6)), src.height)
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Geometry does not overlap raster")
    xmin, ymin, xmax, ymax = window_bounds(
        Window(col0, row0, col1 - col0, row1 - row0), src.transform
    )
    return xmin, ymin, xmax, ymax, abs(src.transform.a), abs(src.transform.e)


def _align_vector_crs(image_path: str, label_path: str, tmp_dir: str) -> str:
    with rasterio.open(image_path) as src:
        image_crs = src.crs
    label_gdf = load_vector_mask(label_path)
    if image_crs is None or label_gdf.crs == image_crs:
        return label_path
    src_crs = label_gdf.crs
    extent = label_gdf.attrs.get("extent_geometry")
    label_gdf.to_crs(image_crs, inplace=True)
    if extent is not None and src_crs is not None:
        extent = gpd.GeoSeries([extent], crs=src_crs).to_crs(image_crs).iloc[0]
    out = Path(tmp_dir) / f"{Path(label_path).stem}_aligned.gpkg"
    save_vector_mask(label_gdf, out, extent)
    return str(out)


@with_connection_retry
@log_stage(stage_name="clip_raster_to_geometry", log_memory=True)
def clip_raster_to_geometry(
    image_path: str,
    geometry: box,
    prefix: str,
    tmp_dir: str,
    extent: Tuple[float, float, float, float, float, float],
    t_srs: Optional[str] = None,
    dst_nodata: Optional[Union[int, float]] = None,
    src_nodata: Optional[Union[int, float, str]] = None,
):
    """Clip raster to geometry on the given pixel-aligned extent."""
    if Path(image_path).suffix.lower() == ".vrt":
        vrt_image_path = Path(tmp_dir) / f"{Path(image_path).stem}_vrt.tif"
        gdal_translate_copy(image_path, vrt_image_path)
        source_path = vrt_image_path
        cleanup_vrt = True
    else:
        source_path = image_path
        cleanup_vrt = False
    temp_geom_path = Path(tmp_dir) / f"{prefix}_clip_geom.shp"
    clipped_image_path = Path(tmp_dir) / f"{Path(image_path).stem}_clipped_{prefix}.tif"
    xmin, ymin, xmax, ymax, xres, yres = extent

    try:
        with rasterio.open(source_path) as src:
            crs = src.crs
            spec_nd, spec_src = nodata_spec(src)
        if dst_nodata is None:
            dst_nodata = spec_nd
            if spec_src == "fallback_zero":
                logger.warning(
                    "%s has no nodata; using 0 for clipped output", image_path
                )
        if src_nodata is None:
            src_nodata = dst_nodata

        cutline_crs = t_srs or crs
        if hasattr(geometry, "geom_type"):
            gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geometry], crs=cutline_crs)
        else:
            gdf = gpd.GeoDataFrame(
                {"id": list(range(len(geometry)))},
                geometry=list(geometry),
                crs=cutline_crs,
            )
        gdf.to_file(temp_geom_path, driver="ESRI Shapefile")
        del gdf

        cmd = [
            "gdalwarp",
            "-overwrite",
            "-cutline",
            str(temp_geom_path),
            "-te",
            str(xmin),
            str(ymin),
            str(xmax),
            str(ymax),
            "-tr",
            str(xres),
            str(yres),
            "-r",
            "near",
        ]
        if t_srs:
            cmd.extend(["-t_srs", t_srs])
        cmd.extend(
            [
                "-srcnodata",
                str(src_nodata),
                "-dstnodata",
                str(dst_nodata),
                "-of",
                "GTiff",
                "-co",
                "TILED=YES",
                "-co",
                "BLOCKXSIZE=256",
                "-co",
                "BLOCKYSIZE=256",
                str(source_path),
                str(clipped_image_path),
            ]
        )
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return clipped_image_path
    finally:
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            shp_file = temp_geom_path.with_suffix(ext)
            if shp_file.exists():
                shp_file.unlink()
        if cleanup_vrt and Path(source_path).exists():
            Path(source_path).unlink()


@log_stage(stage_name="clip_vector_to_extent", log_memory=True)
def clip_vector_to_extent(label_path: str, geometry: box, tmp_dir: str) -> Path:
    clipped_label_path = Path(tmp_dir) / f"{Path(label_path).stem}_clipped_label.gpkg"
    label = load_vector_mask(label_path)
    extent_gdf = gpd.GeoDataFrame(geometry=[geometry], crs=label.crs)
    clipped_gdf = gpd.clip(label, extent_gdf)
    save_vector_mask(clipped_gdf, clipped_label_path, extent_geometry=geometry)
    return clipped_label_path


@log_stage(stage_name="clip_to_intersection", log_memory=True)
def clip_to_intersection(
    image_path: str, label_path: str, label_type: str, tmp_dir: str
):
    """Clip image and label onto the image pixel grid at their intersection."""
    if label_type == "vector":
        label_path = _align_vector_crs(image_path, label_path, tmp_dir)
    image_bounds, label_bounds = _pair_geoms(image_path, label_path, label_type)
    intersection = label_bounds.intersection(image_bounds)
    if intersection.is_empty:
        return None, None
    with rasterio.open(image_path) as src:
        extent = _pixel_aligned_extent(src, intersection)
        t_srs = src.crs.to_string() if src.crs else None
    clipped_image = clip_raster_to_geometry(
        image_path, intersection, "image", tmp_dir, extent
    )
    if label_type == "raster":
        clipped_label = clip_raster_to_geometry(
            label_path,
            intersection,
            "label",
            tmp_dir,
            extent,
            t_srs=t_srs,
            dst_nodata=IGNORE_INDEX,
            src_nodata=label_src_nodata(label_path),
        )
        clipped_label = apply_image_mask_to_label(
            clipped_image, clipped_label, tmp_dir
        )
    else:
        snapped = box(extent[0], extent[1], extent[2], extent[3])
        clipped_label = clip_vector_to_extent(label_path, snapped, tmp_dir)
    return clipped_image, clipped_label


@log_stage(stage_name="create_nodata_mask", log_memory=True)
def create_nodata_mask(image_path: str) -> Optional[gpd.GeoDataFrame]:
    with rasterio.open(image_path) as src:
        mask_array = np.zeros((src.height, src.width), dtype=np.uint8)
        for _, win in src.block_windows(1):
            row_slice, col_slice = win.toslices()
            mask_array[row_slice, col_slice] = window_valid(src, win).astype("uint8")
        transform = src.transform
        crs = src.crs
    shapes = rasterio.features.shapes(
        mask_array, mask=mask_array > 0, transform=transform
    )
    geometries = [shape(geom) for geom, val in shapes]
    if not geometries:
        return None
    return gpd.GeoDataFrame(geometry=geometries, crs=crs).dissolve()


@log_stage(stage_name="apply_image_mask_to_label", log_memory=True)
def apply_image_mask_to_label(
    image_path: str, label_path: str, tmp_dir: str
) -> Path:
    out = Path(tmp_dir) / f"{Path(label_path).stem}_ign.tif"
    with rasterio.open(image_path) as img, rasterio.open(label_path) as lbl:
        if (img.width, img.height) != (lbl.width, lbl.height):
            raise ValueError("Image and label dimensions must match")
        profile = lbl.profile.copy()
        profile.update(nodata=IGNORE_INDEX)
        with rasterio.open(out, "w", **profile) as dst:
            for _, window in img.block_windows(1):
                valid = window_valid(img, window)
                data = lbl.read(window=window)
                data[:, ~valid] = IGNORE_INDEX
                dst.write(data, window=window)
    return out


@log_stage(stage_name="apply_nodata_mask", log_memory=True)
def apply_nodata_mask(
    label_path: str, nodata_mask: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    This function clips the vector data to the valid area defined by the raster's no-data mask.
    """
    vector_data = load_vector_mask(label_path)
    if nodata_mask is None:
        return vector_data
    if nodata_mask.crs != vector_data.crs:
        nodata_mask = nodata_mask.to_crs(vector_data.crs)
    res = gpd.overlay(vector_data, nodata_mask, how="intersection")
    del vector_data, nodata_mask
    return res


def _compute_erosion_dist(
    pixel_size: float,
    target_gap_m: float | None,
    max_gsd_for_erosion: float,
) -> float:
    """Returns per-polygon inward erosion distance in CRS units.

    Uses a sensor-adaptive formula when target_gap_m is not specified:
    guarantees a ≥2-pixel gap between adjacent instances, capped at 0.6m
    per side (1.2m total) to avoid collapsing typical building footprints.
    Returns 0.0 for coarse sensors where erosion is counterproductive.
    """
    if pixel_size > max_gsd_for_erosion:
        return 0.0
    if target_gap_m is not None:
        return target_gap_m / 2.0
    return min(pixel_size, 0.6)


@log_stage(stage_name="rasterize_vector", log_memory=True)
def rasterize_vector(
    vector: gpd.GeoDataFrame,
    image_path: str,
    label_name: str,
    tmp_dir: str,
    attr_field: List[str] = None,
    attr_values: list = None,
    continuous: bool = True,
    default_burn_value: int = 1,
    dtype: str = "uint8",
    erosion_classes: List[str] = None,
    target_gap_m: float = None,
    max_gsd_for_erosion: float = 1.0,
    min_erosion_area_m2: float = 10.0,
    valid_area: gpd.GeoDataFrame | None = None,
) -> str:
    """Rasterize vector data to a raster.

    Args:
        erosion_classes: Subset of attr_values whose geometries should be
            eroded inward before rasterization (e.g. ["Building"]).  Only
            meaningful when attr_field/attr_values are provided.  Pass None
            to disable erosion entirely.
        target_gap_m: Desired total physical gap (metres) to enforce between
            adjacent eroded instances.  None uses the sensor-adaptive formula
            (≥2-pixel gap, capped at 1.2 m total).
        max_gsd_for_erosion: Sensors with GSD above this value (metres) skip
            erosion — gaps are sub-pixel and erosion only shrinks footprints.
        min_erosion_area_m2: Eroded geometries whose area falls below this
            threshold are restored to their original shape to prevent small
            structures (sheds, garages) from collapsing.
    """
    temp_vector_path = Path(tmp_dir) / f"{label_name}.shp"
    rasterized_label_path = Path(tmp_dir) / f"{label_name}_rasterized.tif"

    try:
        if vector.empty:
            vector_clean = gpd.GeoDataFrame(
                {"burn_val": pd.Series(dtype=float)},
                geometry=[],
                crs=vector.crs,
            )
            burn_attribute = "burn_val"
            erosion_burn_vals: set = set()
        else:
            vector_clean = vector[
                ~vector.geometry.is_empty & vector.geometry.notnull()
            ].copy()
            if attr_field and attr_values and not vector_clean.empty:
                resolved = resolve_attr_field(vector_clean.columns, attr_field)
                if resolved is None:
                    raise ValueError(
                        f"None of the requested attr_field(s) {attr_field} found in "
                        f"vector columns {list(vector_clean.columns)}"
                    )
                attr_field = resolved
                cont_vals_dict = {
                    src: (dst + 1 if continuous else src)
                    for dst, src in enumerate(attr_values)
                }
                cont_vals_dict.update(
                    {str(k): v for k, v in list(cont_vals_dict.items())}
                )
                vector_clean["burn_val"] = vector_clean[attr_field].map(cont_vals_dict)
                vector_clean = vector_clean.dropna(subset=["burn_val"])
                erosion_burn_vals = {
                    cont_vals_dict.get(str(v) if isinstance(v, str) else v)
                    for v in (erosion_classes or [])
                } - {None}
            else:
                if not vector_clean.empty:
                    vector_clean["burn_val"] = default_burn_value
                erosion_burn_vals = set()
            burn_attribute = "burn_val"

        with rasterio.open(image_path) as src:
            transform = src.transform
            src_width = src.width
            src_height = src.height

        if transform == rasterio.Affine.identity():
            transform = rasterio.transform.from_origin(0, 0, 1, 1)

        pixel_size = min(transform.a, -transform.e)
        erosion_dist = _compute_erosion_dist(
            pixel_size, target_gap_m, max_gsd_for_erosion
        )
        min_width_m = 1.5

        if (
            erosion_dist > 0
            and erosion_burn_vals
            and not vector_clean.empty
            and "burn_val" in vector_clean.columns
        ):
            erode_mask = vector_clean["burn_val"].isin(erosion_burn_vals)
            if erode_mask.any():
                orig = vector_clean.loc[erode_mask, "geometry"]
                too_small_to_erode = orig.area < (min_erosion_area_m2 * 3.0)
                eroded = orig.buffer(-erosion_dist)
                eroded_bounds = eroded.bounds
                eroded_width = (eroded_bounds["maxx"] - eroded_bounds["minx"]).combine(
                    eroded_bounds["maxy"] - eroded_bounds["miny"], min
                )
                restore = (
                    eroded.is_empty
                    | (eroded.area < min_erosion_area_m2)
                    | (eroded_width < min_width_m)
                    | too_small_to_erode
                )
                vector_clean.loc[erode_mask, "geometry"] = eroded.where(~restore, orig)

        vector_clean = vector_clean[~vector_clean.geometry.is_empty].copy()
        eroded_building_gdf = (
            vector_clean[vector_clean["burn_val"].isin(erosion_burn_vals)].copy()
            if not vector_clean.empty and erosion_burn_vals
            else gpd.GeoDataFrame()
        )

        crs = vector_clean.crs
        if valid_area is not None and not valid_area.empty:
            geom = valid_area.geometry.unary_union
            if geom is not None and not geom.is_empty:
                if crs is None:
                    crs = valid_area.crs
                elif valid_area.crs is not None and valid_area.crs != crs:
                    geom = (
                        gpd.GeoSeries([geom], crs=valid_area.crs).to_crs(crs).iloc[0]
                    )
                valid_row = gpd.GeoDataFrame(
                    {"burn_val": [0]}, geometry=[geom], crs=crs
                )
                vector_clean = gpd.GeoDataFrame(
                    pd.concat([valid_row, vector_clean], ignore_index=True),
                    geometry="geometry",
                    crs=crs,
                )
        if vector_clean.empty:
            return None, gpd.GeoDataFrame()
        vector_clean = vector_clean.sort_values("burn_val")
        vector_clean.to_file(temp_vector_path, driver="ESRI Shapefile")
        del vector_clean

        xmin = str(transform.c)
        ymin = str(transform.f + src_height * transform.e)
        xmax = str(transform.c + src_width * transform.a)
        ymax = str(transform.f)
        xres, yres = str(transform.a), str(-transform.e)

        mapping = {
            "uint8": "Byte",
            "uint16": "UInt16",
            "int16": "Int16",
            "uint32": "UInt32",
            "int32": "Int32",
            "float32": "Float32",
            "float64": "Float64",
        }

        cmd = [
            "gdal_rasterize",
            "-a",
            burn_attribute,
            "-a_nodata",
            str(IGNORE_INDEX),
            "-tr",
            xres,
            yres,
            "-te",
            xmin,
            ymin,
            xmax,
            ymax,
            "-ot",
            mapping.get(dtype, "Byte"),
            "-of",
            "GTiff",
            "-init",
            str(IGNORE_INDEX),
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=256",
            "-co",
            "BLOCKYSIZE=256",
            str(temp_vector_path),
            str(rasterized_label_path),
        ]

        subprocess.run(cmd, check=True)

        return rasterized_label_path, eroded_building_gdf
    finally:
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            shp_file = temp_vector_path.with_suffix(ext)
            if shp_file.exists():
                shp_file.unlink()


@log_stage(stage_name="prepare_vector_labels", log_memory=True)
def prepare_vector_labels(
    label_path: str,
    image_path: str,
    tmp_dir: str,
    attr_field: List[str] = None,
    attr_values: list = None,
    erosion_classes: List[str] = None,
    target_gap_m: float = None,
    max_gsd_for_erosion: float = 1.0,
    min_erosion_area_m2: float = 5.0,
    building_class_val: int | None = None,
    road_class_val: int | None = None,
    max_gsd_for_road_targets: float = 1.0,
):
    """Prepares vector labels for tiling."""
    nodata_mask_gdf = create_nodata_mask(image_path)
    label_gdf = apply_nodata_mask(label_path, nodata_mask_gdf)
    label_name = Path(label_path).stem
    rasterized_label_path, eroded_building_gdf = rasterize_vector(
        label_gdf,
        image_path,
        label_name,
        tmp_dir,
        attr_field,
        attr_values,
        erosion_classes=erosion_classes,
        target_gap_m=target_gap_m,
        max_gsd_for_erosion=max_gsd_for_erosion,
        min_erosion_area_m2=min_erosion_area_m2,
        valid_area=nodata_mask_gdf,
    )
    targets_paths = {}
    field = (
        resolve_attr_field(label_gdf.columns, attr_field)
        if attr_field and attr_values
        else None
    )

    if building_class_val is not None:
        if not eroded_building_gdf.empty:
            build_gdf = eroded_building_gdf
        elif field:
            build_gdf = label_gdf[
                label_gdf[field].astype(str) == str(building_class_val)
            ]
        else:
            build_gdf = gpd.GeoDataFrame()

        if not build_gdf.empty:
            targets_paths.update(
                compute_building_targets(
                    build_gdf, image_path, tmp_dir, label_name
                )
            )

    if road_class_val is not None:
        if field:
            road_gdf = label_gdf[
                label_gdf[field].astype(str) == str(road_class_val)
            ]
        else:
            road_gdf = gpd.GeoDataFrame()

        if not road_gdf.empty:
            targets_paths.update(
                compute_road_targets(
                    road_gdf,
                    image_path,
                    tmp_dir,
                    label_name,
                    max_gsd_for_targets=max_gsd_for_road_targets,
                )
            )

    del nodata_mask_gdf
    return rasterized_label_path, targets_paths, label_gdf
