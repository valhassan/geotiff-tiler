import functools
import gc
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import fiona
import geopandas as gpd
import numpy as np
import psutil
import pystac
import rasterio
from shapely.geometry import box, shape

from .checks import (
    check_alignment,
    check_image_validity,
    check_label_type,
    check_label_validity,
    check_stac,
    is_image_georeferenced,
    is_label_georeferenced,
)
from .geoutils import select_bands, stack_bands, with_connection_retry
from .stacitem import SingleBandItemEO

logger = logging.getLogger(__name__)


def gdal_translate_copy(src, dst):
    result = subprocess.run(
        ["gdal_translate", "-of", "GTiff", src, dst], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"GDAL failed: {result.stderr}")


def log_stage(stage_name=None, log_memory=False, force_gc=True):
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


def load_vector_mask(
    mask_path: str, skip_layer: str | None = "extent"
) -> gpd.GeoDataFrame:
    """Loads a vector mask from a path."""
    layers = fiona.listlayers(mask_path)
    extent_layer = next((layer for layer in layers if "extent" in layer.lower()), None)
    main_layer = next(
        (layer for layer in layers if skip_layer not in layer.lower()), None
    )
    if main_layer is None:
        raise ValueError(f"No suitable layer found in {mask_path}")
    result = gpd.read_file(mask_path, layer=main_layer)
    if extent_layer:
        extent_gdf = gpd.read_file(mask_path, layer=extent_layer)
        if not extent_gdf.empty:
            result.attrs["extent_geometry"] = extent_gdf.geometry.iloc[0]
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


@with_connection_retry
@log_stage(stage_name="validate_image", force_gc=False)
def validate_image(
    image_path: str,
    bands_requested: Sequence = ["red", "green", "blue"],
    band_indices: Sequence | None = None,
):
    """Validates an image from a path or stac item"""
    image_asset = None
    if check_stac(image_path):
        item = SingleBandItemEO(
            item=pystac.Item.from_file(image_path), bands_requested=bands_requested
        )
        stac_bands = [value["meta"].href for value in item.bands_requested.values()]
        image_asset = stack_bands(stac_bands)
        return image_asset
    elif Path(image_path).exists():
        if band_indices:
            image_asset = select_bands(image_path, band_indices)
        else:
            image_asset = image_path
        return image_asset
    else:
        raise FileNotFoundError(f"File not found: {image_path}")


def validate_mask(mask_path: str):
    """Validates a mask from a path"""
    if Path(mask_path).exists():
        label_type = check_label_type(mask_path)
        return mask_path, label_type
    else:
        raise FileNotFoundError(f"File not found: {mask_path}")


@log_stage(stage_name="validate_pair", force_gc=False)
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


@log_stage(stage_name="ensure_crs_match", log_memory=True)
def ensure_crs_match(
    image_path: str, label_path: str, label_type: str, tmp_dir: str
) -> Tuple[str, str]:
    """
    Ensure the CRS between image and label match by converting label to match the image.

    Args:
        image_path: Path to the image
        label_path: Path to the label
        label_type: Type of label (raster or vector)
    Returns:
        Tuple of (image_path, aligned_label_path)
    """

    with rasterio.open(image_path) as src_image:
        image_crs = src_image.crs
    if label_type == "raster":
        with rasterio.open(label_path) as src_label:
            if image_crs == src_label.crs:
                logger.info("label crs matches image crs")
                return image_path, label_path
            else:
                aligned_label_path = (
                    Path(tmp_dir) / f"{Path(label_path).stem}_aligned.tif"
                )
                dst_transform, dst_width, dst_height = (
                    rasterio.warp.calculate_default_transform(
                        src_label.crs,
                        image_crs,
                        src_label.width,
                        src_label.height,
                        *src_label.bounds,
                    )
                )
                dst_kwargs = src_label.meta.copy()
                dst_kwargs.update(
                    {
                        "crs": image_crs,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height,
                        "driver": "GTiff",
                    }
                )
                with rasterio.open(aligned_label_path, "w", **dst_kwargs) as dst_label:
                    for i in range(1, src_label.count + 1):
                        rasterio.warp.reproject(
                            source=rasterio.band(src_label, i),
                            destination=rasterio.band(dst_label, i),
                            src_transform=src_label.transform,
                            src_crs=src_label.crs,
                            dst_transform=dst_transform,
                            dst_crs=image_crs,
                            resampling=rasterio.warp.Resampling.nearest,
                        )
                logger.info("label crs does not match image crs, reprojecting label")
                return image_path, aligned_label_path
    elif label_type == "vector":
        label_gdf = load_vector_mask(label_path)
        if image_crs == label_gdf.crs:
            logger.info("label crs matches image crs")
            del label_gdf
            return image_path, label_path
        else:
            aligned_label_path = Path(tmp_dir) / f"{Path(label_path).stem}_aligned.gpkg"
            label_gdf.to_crs(image_crs, inplace=True)
            extent_geometry = label_gdf.attrs.get("extent_geometry")
            save_vector_mask(label_gdf, aligned_label_path, extent_geometry)
            logger.info("label crs does not match image crs, reprojecting label")
            del label_gdf
            return image_path, aligned_label_path


@log_stage(stage_name="calculate_overlap", log_memory=True)
def calculate_overlap(
    image_path: str, label_path: str, label_type: str
) -> Tuple[float, str]:
    """
    Calculate the overlap between image and label data.

    Args:
        image: Opened rasterio dataset
        label: Either a rasterio dataset or GeoDataFrame

    Returns:
        Tuple of (overlap_percentage, message)
    """
    # try:
    # Get image bounds as a box
    with rasterio.open(image_path) as image:
        image_bounds = box(*image.bounds)
    if label_type == "raster":
        with rasterio.open(label_path) as label:
            label_bounds = box(*label.bounds)
    elif label_type == "vector":
        label = load_vector_mask(label_path)
        if hasattr(label, "attrs") and "extent_geometry" in label.attrs:
            label_bounds = label.attrs["extent_geometry"]
        else:
            label_bounds = box(*label.total_bounds)

    # Calculate intersection and union areas
    intersection_area = image_bounds.intersection(label_bounds).area
    union_area = image_bounds.union(label_bounds).area

    if union_area == 0:
        return 0.0, "No valid area found"

    overlap_percentage = (intersection_area / union_area) * 100

    if overlap_percentage == 0:
        return 0.0, "No overlap between image and label"
    del image, label
    return overlap_percentage, f"Overlap percentage: {overlap_percentage:.2f}%"

    # except Exception as e:
    #     return 0.0, f"Error calculating overlap: {str(e)}"


@log_stage(stage_name="get_intersection", log_memory=True)
def get_intersection(image_path: str, label_path: str, label_type: str) -> box:
    """
    Find the intersection area between an image and label (raster or vector).

    Args:
        image: Opened rasterio dataset for the image
        label: Either a rasterio dataset or GeoDataFrame for the label

    Returns:
        shapely.geometry.box representing the intersection area
    """
    with rasterio.open(image_path) as image:
        image_bounds = box(*image.bounds)
    if label_type == "raster":
        with rasterio.open(label_path) as label:
            label_bounds = box(*label.bounds)
    elif label_type == "vector":
        label = load_vector_mask(label_path)
        if hasattr(label, "attrs") and "extent_geometry" in label.attrs:
            label_bounds = label.attrs["extent_geometry"]
        else:
            label_bounds = box(*label.total_bounds)
    intersection = label_bounds.intersection(image_bounds)
    if intersection.is_empty:
        return None
    return intersection


@with_connection_retry
@log_stage(stage_name="clip_raster_to_geometry", log_memory=True)
def clip_raster_to_geometry(image_path: str, geometry: box, prefix: str, tmp_dir: str):
    """
    Clip raster to exact geometry.
    """
    if Path(image_path).suffix.lower() == ".vrt":
        vrt_image_path = Path(tmp_dir) / f"{Path(image_path).stem}_vrt.tif"
        gdal_translate_copy(image_path, vrt_image_path)
        with rasterio.open(vrt_image_path) as src:
            nodata = src.nodata
            crs = src.crs
        source_path = vrt_image_path
        cleanup_vrt = True
    else:
        source_path = image_path
        cleanup_vrt = False
        with rasterio.open(image_path) as src:
            nodata = src.nodata
            crs = src.crs
    temp_geom_path = Path(tmp_dir) / f"{prefix}_clip_geom.shp"
    clipped_image_path = Path(tmp_dir) / f"{Path(image_path).stem}_clipped_{prefix}.tif"

    try:
        if hasattr(geometry, "geom_type"):
            gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geometry], crs=crs)
        else:
            gdf = gpd.GeoDataFrame(
                {"id": list(range(len(geometry)))}, geometry=list(geometry), crs=crs
            )

        gdf.to_file(temp_geom_path, driver="ESRI Shapefile")
        del gdf

        cmd = [
            "gdalwarp",
            "-cutline",
            str(temp_geom_path),
            "-crop_to_cutline",
            "-dstnodata",
            str(nodata),
            "-of",
            "GTiff",
            str(source_path),
            str(clipped_image_path),
        ]
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
def clip_vector_to_extent(
    label_path: str, geometry: box, tmp_dir: str
) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to the given extent.

    Args:
        gdf: Input GeoDataFrame
        extent: shapely box of the target extent

    Returns:
        Clipped GeoDataFrame
    """
    clipped_label_path = Path(tmp_dir) / f"{Path(label_path).stem}_clipped_label.gpkg"
    label = load_vector_mask(label_path)
    extent_gdf = gpd.GeoDataFrame(geometry=[geometry], crs=label.crs)
    clipped_gdf = gpd.clip(label, extent_gdf)
    save_vector_mask(clipped_gdf, clipped_label_path, extent_geometry=geometry)
    # del label, extent_gdf, clipped_gdf, geometry
    return clipped_label_path


@log_stage(stage_name="clip_to_intersection", log_memory=True)
def clip_to_intersection(
    image_path: str, label_path: str, label_type: str, tmp_dir: str
):
    """
    Clip an image and label to the intersection of the two.
    """
    intersection = get_intersection(image_path, label_path, label_type)
    if intersection is None:
        return None, None
    clipped_image_path = clip_raster_to_geometry(
        image_path, intersection, "image", tmp_dir
    )
    if label_type == "raster":
        clipped_label_path = clip_raster_to_geometry(
            label_path, intersection, "label", tmp_dir
        )
    elif label_type == "vector":
        clipped_label_path = clip_vector_to_extent(label_path, intersection, tmp_dir)
    del intersection
    return clipped_image_path, clipped_label_path


@log_stage(stage_name="create_nodata_mask", log_memory=True)
def create_nodata_mask(
    image_path: str, nodata_value: Optional[Union[int, float]] = None
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a vector mask from raster nodata values.
    """
    with rasterio.open(image_path) as src_image:
        if nodata_value is None:
            nodata_value = src_image.nodata
            if not isinstance(nodata_value, (int, float)):
                return None
        data = src_image.read()
        nodata_mask = data != nodata_value
        nodata_mask_flat = np.any(nodata_mask, axis=0)
        mask_array = nodata_mask_flat.astype("uint8")
        transform = src_image.transform
        crs = src_image.crs
        shapes = rasterio.features.shapes(
            mask_array, mask=mask_array > 0, transform=transform
        )
        geometries = [shape(geom) for geom, val in shapes]
        if not geometries:
            return None
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        gdf = gdf.dissolve()
    del (
        data,
        nodata_mask,
        nodata_mask_flat,
        mask_array,
        transform,
        crs,
        shapes,
        geometries,
    )
    return gdf


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
) -> str:
    """
    Rasterize vector data to a raster.
    """
    if vector.empty:
        return None

    temp_vector_path = Path(tmp_dir) / f"{label_name}.shp"
    rasterized_label_path = Path(tmp_dir) / f"{label_name}_rasterized.tif"

    try:
        vector_clean = vector[
            ~vector.geometry.is_empty & vector.geometry.notnull()
        ].copy()

        if attr_field and attr_values:
            field_id = set(attr_field)
            attr_field = field_id.intersection(vector_clean.columns).pop()
            cont_vals_dict = {
                src: (dst + 1 if continuous else src)
                for dst, src in enumerate(attr_values)
            }

            if all(
                isinstance(val, str)
                for val in vector_clean[attr_field].unique().tolist()
            ):
                cont_vals_dict = {str(src): dst for src, dst in cont_vals_dict.items()}

            vector_clean["burn_val"] = vector_clean[attr_field].map(cont_vals_dict)
            vector_clean = vector_clean.sort_values("burn_val")
            burn_attribute = "burn_val"
        else:
            vector_clean["burn_val"] = default_burn_value
            burn_attribute = "burn_val"

        vector_clean.to_file(temp_vector_path, driver="ESRI Shapefile")
        del vector_clean

        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            transform = src.transform

        if transform is rasterio.Affine.identity():
            transform = rasterio.transform.from_origin(0, 0, 1, 1)

        xmin = str(transform.c)
        ymin = str(transform.f + src.height * transform.e)
        xmax = str(transform.c + src.width * transform.a)
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
            "255",
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
            "0",
            "-at",
            str(temp_vector_path),
            str(rasterized_label_path),
        ]

        subprocess.run(cmd, check=True)

        return rasterized_label_path
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
):
    """Prepares vector labels for tiling"""
    nodata_mask_gdf = create_nodata_mask(image_path)
    label_gdf = apply_nodata_mask(label_path, nodata_mask_gdf)
    label_name = Path(label_path).stem
    rasterized_label_path = rasterize_vector(
        label_gdf, image_path, label_name, tmp_dir, attr_field, attr_values
    )
    del nodata_mask_gdf, label_gdf
    return rasterized_label_path
