"""STAC, CRS checks, band VRTs, and vector clip/export."""

from __future__ import annotations

import functools
import json
import logging
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pystac
import rasterio
import shapely
from affine import Affine
from pystac.extensions.eo import Band, ItemEOExtension
from rasterio import MemoryFile
from rasterio.shutil import copy as riocopy
from rasterio.windows import Window, bounds as window_bounds
from requests.exceptions import ConnectionError, RequestException, Timeout
from shapely.geometry import box

logger = logging.getLogger(__name__)

def check_label_type(label_path: str) -> str:
    """Return ``raster`` or ``vector`` from the label file extension."""
    lower = str(label_path).lower()
    if lower.endswith((".tif", ".tiff")):
        return "raster"
    if lower.endswith((".geojson", ".gpkg", ".shp")):
        return "vector"
    raise ValueError(
        f"Invalid label type: {label_path}, "
        "must be raster (.tif/.tiff) or vector (.geojson/.gpkg/.shp)"
    )


def is_image_georeferenced(image: rasterio.DatasetReader) -> bool:
    """Checks if the image is georeferenced"""
    return image.crs is not None and image.transform is not None


def is_label_georeferenced(
    label: Union[rasterio.DatasetReader, gpd.GeoDataFrame],
) -> bool:
    """Checks if the label is georeferenced"""
    if isinstance(label, rasterio.DatasetReader):
        return is_image_georeferenced(label)
    elif isinstance(label, gpd.GeoDataFrame):
        return label.crs is not None
    else:
        return False


def check_alignment(
    image: rasterio.DatasetReader, label: rasterio.DatasetReader
) -> bool:
    """Checks if the image and label are aligned"""
    dims_match = (image.width == label.width) and (image.height == label.height)

    return dims_match


def check_image_validity(image: rasterio.DatasetReader) -> Tuple[bool, str]:
    """Check if the image has positive dimensions."""
    if image.width <= 0 or image.height <= 0:
        return False, "Invalid dimensions"
    return True, "Image is valid"


def check_label_validity(
    label: Union[rasterio.DatasetReader, gpd.GeoDataFrame],
) -> Tuple[bool, str]:
    """
    Check if the label data is valid.

    Args:
        label: Either a rasterio dataset or GeoDataFrame

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if isinstance(label, rasterio.DatasetReader):
            # Check raster label
            if label.width <= 0 or label.height <= 0:
                return False, "Invalid dimensions"
            return True, "Label is valid"

        elif isinstance(label, gpd.GeoDataFrame):
            # Check vector label
            if label.empty:
                return False, "Label vector is empty"

            if not label.geometry.is_valid.all():
                return False, "Label vector contains invalid geometries"

        else:
            return False, f"Unsupported label type: {type(label)}"

        return True, "Label is valid"

    except Exception as e:
        return False, f"Error reading label: {str(e)}"


def with_connection_retry(func):
    """
    Decorator to add connection retry capability to functions accessing remote resources.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract retry parameters from kwargs if provided, with enhanced defaults for STAC
        max_retries = kwargs.pop("max_retries", 3)
        retry_delay = kwargs.pop("retry_delay", 2.0)
        kwargs.pop("timeout", None)
        
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
                
            except (ConnectionError, Timeout, RequestException) as e:
                # Original connection errors
                retry_count += 1
                last_error = e
                error_msg = str(e).lower()
                logger.warning(f"Network error in {func.__name__}. "
                              f"Retry {retry_count}/{max_retries}. Error: {str(e)}")
                
            except rasterio.errors.RasterioIOError as e:
                # Rasterio-specific errors (often network-related for remote files)
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    "connection", "timeout", "remote", "http", "ssl", "tls",
                    "network", "unreachable", "refused", "reset", "broken pipe"
                ]):
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Rasterio network error in {func.__name__}. "
                                  f"Retry {retry_count}/{max_retries}. Error: {str(e)}")
                else:
                    # Non-network rasterio errors should not be retried
                    logger.error(f"Rasterio error in {func.__name__} (not retrying): {str(e)}")
                    raise
                    
            except Exception as e:
                # Check if it's a network-related error in disguise
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    "connection", "timeout", "network", "dns", "resolve",
                    "unreachable", "refused", "reset", "broken pipe", "ssl", "tls"
                ]):
                    retry_count += 1
                    last_error = e
                    logger.warning(f"Network-related error in {func.__name__}. "
                                  f"Retry {retry_count}/{max_retries}. Error: {str(e)}")
                else:
                    # Non-network errors should be raised immediately
                    logger.error(f"Non-network error in {func.__name__}: {str(e)}")
                    raise
            
            # If we reach here, we're retrying
            if retry_count < max_retries:
                # Exponential backoff with jitter
                delay = retry_delay * (2 ** (retry_count - 1)) + (retry_count * 0.1)
                logger.info(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            
        # All retries exhausted
        if last_error:
            logger.error(f"Failed after {max_retries} retries in {func.__name__}")
            raise ConnectionError(f"Failed connection after {max_retries} retries: {str(last_error)}")
            
    return wrapper

@with_connection_retry
def stack_bands(srcs: List, band: int = 1):
    """
    Stacks multiple single-band rasters into a single multiband virtual raster.

    Source:
    https://gis.stackexchange.com/questions/392695/is-it-possible-to-build-a-vrt-file-from-multiple-files-with-rasterio

    Args:
        srcs (List[str]): List of paths/URLs to single-band rasters.
        band (int): Index of band from source raster to stack into multiband VRT(index starts at 1 per GDAL convention).

    Returns:
        str: VRT as a string.
    """
    vrt_bands = []
    for srcnum, src in enumerate(srcs, start=1):
        with rasterio.open(src) as ras, MemoryFile() as mem:
            riocopy(ras, mem.name, driver='VRT')
            vrt_xml = mem.read().decode('utf-8')
            vrt_dataset = ET.fromstring(vrt_xml)
            for bandnum, vrt_band in enumerate(vrt_dataset.iter('VRTRasterBand'), start=1):
                if bandnum == band:
                    vrt_band.set('band', str(srcnum))
                    vrt_bands.append(vrt_band)
                    vrt_dataset.remove(vrt_band)
    for vrt_band in vrt_bands:
        vrt_dataset.append(vrt_band)

    return ET.tostring(vrt_dataset).decode('UTF-8')

def select_bands(src: str, band_indices: Sequence, out_path: str | Path) -> str:
    """Write a VRT with a subset/reorder of bands. Indices are 1-based."""
    src_abs = str(Path(src).resolve()) if Path(src).exists() else str(src)
    with rasterio.open(src_abs) as ras, MemoryFile() as mem:
        riocopy(ras, mem.name, driver="VRT")
        vrt_dataset = ET.fromstring(mem.read().decode("utf-8"))
        by_idx = {
            int(band.get("band")): band
            for band in vrt_dataset.iter("VRTRasterBand")
        }
        for band in list(by_idx.values()):
            vrt_dataset.remove(band)
        nsrc = max(by_idx, default=0)
        for dest_idx, src_idx in enumerate(band_indices, start=1):
            if src_idx not in by_idx:
                raise ValueError(
                    f"band index {src_idx} not in 1..{nsrc} for {src}"
                )
            vrt_band = deepcopy(by_idx[src_idx])
            vrt_band.set("band", str(dest_idx))
            vrt_dataset.append(vrt_band)
        for fn in vrt_dataset.iter("SourceFilename"):
            fn.set("relativeToVRT", "0")
            fn.text = src_abs

    out = Path(out_path)
    out.write_text(ET.tostring(vrt_dataset, encoding="unicode"), encoding="utf-8")
    return str(out)


class SingleBandItemEO(ItemEOExtension):
    """
    Single-Band Stac Item with assets by common name.
    For info on common names, see https://github.com/stac-extensions/eo#common-band-names
    """
    def __init__(
            self,
            item: pystac.Item,
            bands_requested: Optional[Sequence] = None,
    ):
        """

        @param item:
            Stac item containing metadata linking imagery assets
        @param bands_requested:
            band selection which must be a list of STAC Item common names from eo extension.
            See: https://github.com/stac-extensions/eo/#common-band-names
        """
        super().__init__(item)
        self.item = item
        self._assets_by_common_name = None

        if not bands_requested:
            raise ValueError("At least one band should be chosen if assets need to be reached")

        # Create band inventory (all available bands)
        self.bands_all = [band for band in self.asset_by_common_name.keys()]

        # Make sure desired bands are subset of inventory
        if not set(bands_requested).issubset(set(self.bands_all)):
            raise ValueError(f"Requested bands ({bands_requested}) should be a subset of available bands ({self.bands_all})")

        # Filter only requested bands
        self.bands_requested = {band: self.asset_by_common_name[band] for band in bands_requested}
        logging.debug(self.bands_all)
        logging.debug(self.bands_requested)

        bands = []
        for band in self.bands_requested.keys():
            eo_band = self.bands_requested[band]["meta"].extra_fields["eo:bands"][0]
            band = Band.create(
                name=self.bands_requested[band]['name'],
                common_name=band,
                description=self.bands_requested[band]['meta'].description,
                center_wavelength=eo_band.get("center_wavelength"),
                full_width_half_max=eo_band.get("full_width_half_max"))
            bands.append(band)
        self.bands = bands

    @property
    def asset_by_common_name(self) -> Dict:
        """
        Get assets by common band name (only works for assets containing 1 band)
        Adapted from:
        https://github.com/sat-utils/sat-stac/blob/40e60f225ac3ed9d89b45fe564c8c5f33fdee7e8/satstac/item.py#L75
        @return:
        """
        if self._assets_by_common_name is None:
            self._assets_by_common_name = OrderedDict()
            for name, a_meta in self.item.assets.items():
                bands = []
                if 'eo:bands' in a_meta.extra_fields.keys():
                    bands = a_meta.extra_fields['eo:bands']
                if len(bands) == 1:
                    eo_band = bands[0]
                    if 'common_name' in eo_band.keys():
                        common_name = eo_band['common_name']
                        if not self.is_valid_cname(common_name):
                            raise ValueError(f'Must be one of the accepted common names. Got "{common_name}".')
                        else:
                            self._assets_by_common_name[common_name] = {'meta': a_meta, 'name': name}
        if not self._assets_by_common_name:
            raise ValueError("Common names for assets cannot be retrieved")
        return self._assets_by_common_name

    @staticmethod
    def is_valid_cname(common_name: str) -> bool:
        """Checks if a band name is a valid common name according to STAC spec"""
        return True if Band.band_range(common_name) else False


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

    patch_box = box(*window_bounds(window, src_transform))
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

    patch_box_geo = box(*window_bounds(window, src_transform))
    prop_cols = [c for c in label_gdf.columns if c not in drop_cols]

    features = []
    for idx in range(len(label_gdf)):
        geom = label_gdf.geometry.iloc[idx]
        if not geom.is_valid:
            geom = geom.make_valid()
        geom = geom.intersection(patch_box_geo)
        if geom.is_empty:
            continue
        is_truncated = not patch_box_geo.contains(geom)
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
