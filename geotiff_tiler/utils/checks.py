import logging
import pystac
import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from typing import Union, Tuple

logger = logging.getLogger(__name__)

def check_stac(image_path: str) -> bool:
    """Checks if an input string or object is a valid stac item"""
    if isinstance(image_path, pystac.Item):
        return True
    else:
        try:
            pystac.Item.from_file(str(image_path))
            return True
        except Exception:
            return False

def check_label_type(label_path: str) -> bool:
    """Checks if labels are raster or vector based on file extension"""
    if label_path.endswith(('.tif', '.tiff')):
        return 'raster'
    elif label_path.endswith(('.geojson', '.gpkg')):
        return 'vector'
    else:
        raise ValueError(f"Invalid label type: {label_path}, "
                         "must be a raster (.tif, .tiff) or vector (.geojson, .gpkg) file")       

def is_image_georeferenced(image: rasterio.DatasetReader) -> bool:
    """Checks if the image is georeferenced"""
    if image.crs is not None and image.transform is not None:
        return True
    else:
        return False

def is_label_georeferenced(label: Union[rasterio.DatasetReader, gpd.GeoDataFrame]) -> bool:
    """Checks if the label is georeferenced"""
    if isinstance(label, rasterio.DatasetReader):
        return is_image_georeferenced(label)
    elif isinstance(label, gpd.GeoDataFrame):
        return label.crs is not None
    else:
        return False

def check_alignment(image: rasterio.DatasetReader, label: rasterio.DatasetReader) -> bool:
    """Checks if the image and label are aligned"""
    dims_match = (image.width == label.width) and (image.height == label.height)
     
    return dims_match

def check_image_validity(image: rasterio.DatasetReader) -> Tuple[bool, str]:
    """
    Check if the image data is valid.
    
    Args:
        image: Opened rasterio dataset
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check if the image has data
        if image.width <= 0 or image.height <= 0:
            return False, "Invalid dimensions"
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Error reading image: {str(e)}"

def check_label_validity(
    label: Union[rasterio.DatasetReader, gpd.GeoDataFrame]) -> Tuple[bool, str]:
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
                logger.info("Found invalid geometries, fixing with make_valid()")
                label['geometry'] = label.geometry.make_valid()
                return False, "Label vector contains invalid geometries"
                
        else:
            return False, f"Unsupported label type: {type(label)}"
            
        return True, "Label is valid"
        
    except Exception as e:
        return False, f"Error reading label: {str(e)}"