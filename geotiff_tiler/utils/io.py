import time
import logging
import functools
import zarr
import pystac
import rasterio
import fiona
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Sequence
from memory_profiler import profile
from .checks import check_stac, check_label_type
from .stacitem import SingleBandItemEO
from .geoutils import stack_bands, select_bands, with_connection_retry

logger = logging.getLogger(__name__)


@with_connection_retry
def load_image(image_path: str, 
               bands_requested: Sequence = ["red", "green", "blue"], 
               band_indices: Sequence | None = None):
    
    """Loads an image from a path or stac item"""
    
    image_asset = None
    
    if check_stac(image_path):
        item = SingleBandItemEO(item=pystac.Item.from_file(image_path),
                                bands_requested=bands_requested)
        stac_bands = [value['meta'].href for value in item.bands_requested.values()]
        image_asset = stack_bands(stac_bands)
    
    elif Path(image_path).exists():
        if band_indices:
            image_asset = select_bands(image_path, band_indices)
        else:
            image_asset = image_path
    
    if image_asset:
        raster = rasterio.open(image_asset)
        return raster
    else:
        raise FileNotFoundError(f"File not found: {image_path}")

def load_vector(vector: gpd.GeoDataFrame, skip_layer: str | None = None):
    
    pass


def load_mask(mask_path: str, skip_layer: str | None = "extent"):
    """Loads a mask from a path. Accepts a local tiff or json file"""
    if Path(mask_path).exists():
        label_type = check_label_type(mask_path)
        if label_type == 'raster':
            return rasterio.open(mask_path)
        elif label_type == 'vector':
            layers = fiona.listlayers(mask_path)
            extent_layer = next((layer for layer in layers if "extent" in layer.lower()), None)
            main_layer = next(
                (layer for layer in layers if skip_layer not in layer.lower()), None)
            if main_layer is None:
                raise ValueError(f"No suitable layer found in {mask_path}")
            result = gpd.read_file(mask_path, layer=main_layer)
            if extent_layer:
                extent_gdf = gpd.read_file(mask_path, layer=extent_layer)
                if not extent_gdf.empty:
                    result.attrs['extent_geometry'] = extent_gdf.geometry.iloc[0]
            return result
    else:
        raise FileNotFoundError(f"File not found: {mask_path}")

def read_patches_from_zarr(zarr_path, image_name, indices=None):
        """
        Read patches from Zarr store
        
        Parameters:
        -----------
        zarr_path : str or Path
            Path to the Zarr store
        indices : list, slice, or None
            Indices of patches to read. If None, read all patches
            
        Returns:
        --------
        image_patches : np.ndarray
            Image patches with shape (n_patches, channels, height, width)
        label_patches : np.ndarray
            Label patches with shape (n_patches, classes, height, width)
        patch_locations : np.ndarray
            (x, y) coordinates of each patch
        """
        zarr_path = str(zarr_path)
        root = zarr.open(zarr_path, mode='r')
        image_group = root[image_name]
        if indices is None:
            image_patches = image_group['images'][:]
            label_patches = image_group['labels'][:]
            patch_locations = image_group['locations'][:]
        else:
            image_patches = image_group['images'][indices]
            label_patches = image_group['labels'][indices]
            patch_locations = image_group['locations'][indices]
        
        return image_patches, label_patches, patch_locations
    