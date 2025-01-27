import pystac
import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from typing import Union, Tuple

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
        if image.count == 0:
            return False, "Image has no bands"
            
        # Try reading a small sample to verify data accessibility
        window = ((0, min(10, image.height)), (0, min(10, image.width)))
        sample = image.read(1)
        
        # Check if data contains any valid values
        if np.all(sample == image.nodata):
            return False, "Image contains only nodata values"
            
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
            if label.count == 0:
                return False, "Label raster has no bands"
                
            sample = label.read(1, window=((0, min(10, label.height)), 
                                         (0, min(10, label.width))))
            
            if np.all(sample == label.nodata):
                return False, "Label raster contains only nodata values"
                
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

def calculate_overlap(
    image: rasterio.DatasetReader,
    label: Union[rasterio.DatasetReader, gpd.GeoDataFrame]) -> Tuple[float, str]:
    """
    Calculate the overlap between image and label data.
    
    Args:
        image: Opened rasterio dataset
        label: Either a rasterio dataset or GeoDataFrame
        
    Returns:
        Tuple of (overlap_percentage, message)
    """
    try:
        # Get image bounds as a box
        image_bounds = box(*image.bounds)
        print(f"image_bounds: {image_bounds}")
        print(f"image.crs: {image.crs}")
        # print(f"image.transform: {image.transform}")
        # Get label bounds
        if isinstance(label, rasterio.DatasetReader):
            # label_bounds = box(*label.bounds)
            # print(f"label_bounds: {label_bounds}")
            print(f"label.crs: {label.crs}")
            # print(f"transform: {label.transform}")
        else:
            # For GeoDataFrame, ensure CRS matches
            # label_gdf = label.to_crs(image.crs)
            print(f"label.crs: {label.crs}")
            label_bounds = label.total_bounds
            label_bounds = box(*label_bounds)
            print(f"label_bounds: {label_bounds}")
            print(f"label.crs: {label.crs}")
            # print(f"transform: {label.transform}")
        
        # Calculate intersection and union areas
        intersection_area = image_bounds.intersection(label_bounds).area
        union_area = image_bounds.union(label_bounds).area
        
        if union_area == 0:
            return 0.0, "No valid area found"
        
        overlap_percentage = (intersection_area / union_area) * 100
        
        if overlap_percentage == 0:
            return 0.0, "No overlap between image and label"
        
        return overlap_percentage, f"Overlap percentage: {overlap_percentage:.2f}%"
        
    except Exception as e:
        return 0.0, f"Error calculating overlap: {str(e)}" 