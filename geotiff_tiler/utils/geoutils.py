import time
import logging
import rasterio
import functools
import rasterio.features
import rasterio.mask
import xml.etree.ElementTree as ET
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, box
from pathlib import Path
from rasterio.shutil import copy as riocopy
from rasterio import MemoryFile
from rasterio.windows import from_bounds
from requests.exceptions import ConnectionError, Timeout, RequestException
from typing import List, Sequence, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def with_connection_retry(func):
    """
    Decorator to add connection retry capability to functions accessing remote resources.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract retry parameters from kwargs if provided, with enhanced defaults for STAC
        max_retries = kwargs.pop('max_retries', 3) if 'max_retries' in kwargs else 3
        retry_delay = kwargs.pop('retry_delay', 2.0) if 'retry_delay' in kwargs else 2.0  # Increased from 1.0
        timeout = kwargs.pop('timeout', 45.0) if 'timeout' in kwargs else 45.0  # Increased from 30.0
        
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

def select_bands(src: str, band_indices: Optional[Sequence]):
    """Creates a multiband virtual raster containing a subset of all available bands in a source multiband raster.

    Args:
        src (str): Path or URL to a multiband raster.
        band_indices (Sequence, optional): Indices of bands from the source raster to include in the subset
            (indices start at 1 per GDAL convention). Order matters; for example, if the source raster is BGR,
            [3, 2, 1] will create a VRT with bands as RGB.

    Returns:
        str: VRT as a string.
    """
    
    with rasterio.open(src) as ras, MemoryFile() as mem:
        riocopy(ras, mem.name, driver='VRT')
        vrt_xml = mem.read().decode('utf-8')
        vrt_dataset = ET.fromstring(vrt_xml)
        vrt_dataset_dict = {int(band.get('band')): band for band in vrt_dataset.iter("VRTRasterBand")}
        for band in vrt_dataset_dict.values():
            vrt_dataset.remove(band)

        for dest_band_idx, src_band_idx in enumerate(band_indices, start=1):
            vrt_band = vrt_dataset_dict[src_band_idx]
            vrt_band.set('band', str(dest_band_idx))
            vrt_dataset.append(vrt_band)

    return ET.tostring(vrt_dataset).decode('UTF-8')

def ensure_crs_match(image: rasterio.DatasetReader,
                     label: Union[rasterio.DatasetReader, gpd.GeoDataFrame],
                     ) -> Tuple[rasterio.DatasetReader, Union[rasterio.DatasetReader, gpd.GeoDataFrame], bool]:
    """
    Ensure the CRS between image and label match by converting label to match the image.
    
    Args:
        image: Opened rasterio dataset
        label: Either a rasterio dataset or GeoDataFrame
    Returns:
        Tuple of (aligned_image, aligned_label, was_converted)
    """
    # Check if CRS already match
    if image.crs == label.crs:
        return image, label, False
    
    # Convert based on data types
    was_converted = True
    
    if isinstance(label, gpd.GeoDataFrame):
        # For vector labels, conversion is straightforward
        label = label.to_crs(image.crs)
    
    elif isinstance(label, rasterio.DatasetReader):
        # Both are rasters, convert label to match image
        # Important: Using a MemoryFile object that persists beyond this function
        memfile = rasterio.MemoryFile()
        
        # Calculate transform to match the image's pixel grid
        dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            label.crs, image.crs, label.width, label.height, *label.bounds
        )
        
        dst_kwargs = label.meta.copy()
        dst_kwargs.update({
            'crs': image.crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'driver': 'GTiff'  # Ensure driver is set
        })
        
        dst = memfile.open(**dst_kwargs)
        
        # Reproject using nearest neighbor for categorical data
        for i in range(1, label.count + 1):
            rasterio.warp.reproject(
                source=rasterio.band(label, i),
                destination=rasterio.band(dst, i),
                src_transform=label.transform,
                src_crs=label.crs,
                dst_transform=dst_transform,
                dst_crs=image.crs,
                resampling=rasterio.warp.Resampling.nearest  # Correct for categorical data
            )
        
        # Close the original label dataset if it's not needed
        label.close()  # Uncomment if you want to close the original
        
        # Use the reprojected dataset (don't close memfile or dst)
        dst.close()  # Close the destination to flush data
        label = memfile.open()  # Reopen from memfile
    
    return image, label, was_converted

def create_nodata_mask(
    raster: Union[str, Path, rasterio.DatasetReader],
    nodata_value: Optional[Union[int, float]] = None
) -> Optional[gpd.GeoDataFrame]:
    """
    Create a vector mask from raster nodata values.
    
    Args:
        raster: Path to raster file or open rasterio dataset
        nodata_value: Optional override for raster's nodata value
        
    Returns:
        GeoDataFrame containing the nodata mask polygons, or None if no valid 
        nodata value is found
    """
    
    # Get nodata value
    if nodata_value is None:
        nodata_value = raster.nodata
        if not isinstance(nodata_value, (int, float)):
            return None
    
    # Read raster data and create mask
    data = raster.read()
    
    # Create binary mask where True indicates valid data
    nodata_mask = data != nodata_value
    
    # Collapse along band axis - pixel is valid if any band has valid data
    nodata_mask_flat = np.any(nodata_mask, axis=0)
    
    # Convert to uint8 for vectorization
    mask_array = nodata_mask_flat.astype('uint8')
    
    # Get raster metadata for transformation
    transform = raster.transform
    crs = raster.crs
    
    # Vectorize the mask using rasterio.features
    shapes = rasterio.features.shapes(
        mask_array,
        mask=mask_array > 0,  # Only vectorize valid data areas
        transform=transform
    )
    
    # Convert shapes to GeoDataFrame
    geometries = [shape(geom) for geom, val in shapes]
    
    if not geometries:
        return None
        
    gdf = gpd.GeoDataFrame(
        geometry=geometries,
        crs=crs
    )
    
    # Dissolve all polygons into a single geometry
    gdf = gdf.dissolve()
    
    return gdf

def apply_nodata_mask(
    vector_data: Union[gpd.GeoDataFrame],
    nodata_mask: Union[gpd.GeoDataFrame] | None) -> gpd.GeoDataFrame | None:
    
    """
    This function clips the vector data to the valid area defined by the raster's no-data mask.
    
    Args:
        vector_data: GeoDataFrame of vector data
        nodata_mask: GeoDataFrame of nodata mask
        
    Returns:
        Clipped GeoDataFrame
    """
    if nodata_mask is None:
        return vector_data
    
    # Ensure matching CRS
    if nodata_mask.crs != vector_data.crs:
        nodata_mask = nodata_mask.to_crs(vector_data.crs)
        
    # Perform spatial overlay
    return gpd.overlay(vector_data, nodata_mask, how='intersection')

def get_intersection(image: rasterio.DatasetReader, label: Union[rasterio.DatasetReader, gpd.GeoDataFrame]) -> box:
    """
    Find the intersection area between an image and label (raster or vector).
    
    Args:
        image: Opened rasterio dataset for the image
        label: Either a rasterio dataset or GeoDataFrame for the label
        
    Returns:
        shapely.geometry.box representing the intersection area
    """
    # Get image bounds as a box
    image_bounds = box(*image.bounds)
    
    # Get label bounds based on type
    if isinstance(label, rasterio.DatasetReader):
        label_bounds = box(*label.bounds)
    elif isinstance(label, gpd.GeoDataFrame):
        if hasattr(label, 'attrs') and 'extent_geometry' in label.attrs:
            label_bounds = label.attrs['extent_geometry']
        else:
            label_bounds = box(*label.total_bounds)
    else:
        raise ValueError("Label must be either a rasterio.DatasetReader or GeoDataFrame")
    # Find intersection
    intersection = label_bounds.intersection(image_bounds)
    
    if intersection.is_empty:
        return None
    
    return intersection

@with_connection_retry
def clip_raster_to_extent(raster: rasterio.DatasetReader,
                          geometry,
                          write_raster: bool = False,
                          output_path: str = None,
                          ):
    """
    Clip a raster to the given geometry.
    
    Args:
        raster: Input rasterio dataset
        geometry: Shapely geometry or list of geometries to clip by
        write_raster: Whether to write the result to disk
        output_path: Path to save the clipped raster (if write_raster is True)
    Returns:
        Clipped rasterio dataset or path to the saved file
    """
    # Handle single geometry vs list of geometries
    shapes = [geometry] if hasattr(geometry, 'geom_type') else geometry
    nodata_value = raster.nodata
    
    is_vrt = raster.driver.upper() == 'VRT'
    
    if is_vrt:
        with MemoryFile() as memfile:
            temp_meta = raster.meta.copy()
            temp_meta.update({"driver": "GTiff"})
            with memfile.open(**temp_meta) as temp:
                temp.write(raster.read())
                out_image, out_transform = rasterio.mask.mask(temp,
                                                              shapes,
                                                              crop=True,
                                                              all_touched=True,
                                                              nodata=nodata_value)
                out_meta = temp_meta.copy()
    else:
        # Perform the clipping
        out_image, out_transform = rasterio.mask.mask(raster,
                                                      shapes,
                                                      crop=True,
                                                      all_touched=True,
                                                      nodata=nodata_value
                                                      )
        
        out_meta = raster.meta.copy()
    
    out_meta.update({"height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform,
                     "nodata": nodata_value})
    
    if write_raster:
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
        return output_path
    else:
        memfile = MemoryFile()
        with memfile.open(**out_meta) as dest:
            dest.write(out_image)
        return memfile.open()

def clip_vector_to_extent(gdf: gpd.GeoDataFrame, extent: box) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to the given extent.
    
    Args:
        gdf: Input GeoDataFrame
        extent: shapely box of the target extent
        
    Returns:
        Clipped GeoDataFrame
    """
    # Convert extent to GeoDataFrame with same CRS
    extent_gdf = gpd.GeoDataFrame(geometry=[extent], crs=gdf.crs)
    
    # Perform the clip
    clipped_gdf = gpd.clip(gdf, extent_gdf)
    
    return clipped_gdf

def clip_to_intersection(image: rasterio.DatasetReader, 
                         label: Union[rasterio.DatasetReader, gpd.GeoDataFrame], 
                         intersection: box):
    
    """
    Clip image and label to the intersection of the two.
    
    Args:
        image: rasterio.DatasetReader for the image
        label: rasterio.DatasetReader or gpd.GeoDataFrame for the label
        intersection: shapely.geometry.box representing the intersection area
        
    Returns:
        Tuple of clipped image and label
    """
    image_clipped = clip_raster_to_extent(image, intersection)
    if isinstance(label, rasterio.DatasetReader):
        label_clipped = clip_raster_to_extent(label, intersection)
    elif isinstance(label, gpd.GeoDataFrame):
        label_clipped = clip_vector_to_extent(label, intersection)
    else:
        raise ValueError("Label must be either a rasterio.DatasetReader or GeoDataFrame")
    
    return image_clipped, label_clipped

def rasterize_vector(
    vector: gpd.GeoDataFrame,
    image: rasterio.DatasetReader,
    attr_field: List[str] = None,
    attr_values: list = None,
    continuous: bool = True,
    default_burn_value: int = 1,
    dtype: str = "uint8",
    write_raster: bool = False,
    output_path: str = None,
) -> np.ndarray:
    """
    Rasterize a GeoDataFrame with attribute-based filtering and dynamic burn values.

    Args:
        vector: GeoDataFrame containing vector data.
        image: rasterio.DatasetReader object containing the image raster.
        attr_field: Attribute field to filter features.
        attr_values: List of attribute values to retain and map for rasterization.
        continuous: If True, map attribute values to continuous burn values.
        default_burn_value: Default burn value when no attribute filters are applied.
        dtype: Data type of the output raster array.
        write_raster: If True, write the rasterized vector to a file.
        output_path: Path to the output raster file.

    Returns:
        A NumPy array representing the rasterized vector data.
    """
    out_shape = (image.height, image.width)
    if vector.empty:
        return np.zeros(out_shape, dtype=dtype)
    
    transform = image.transform
    if transform is rasterio.Affine.identity():
        transform = rasterio.transform.from_origin(0, 0, 1, 1)
    
    # Determine burn field and values
    if attr_field and attr_values:
        field_id = set(attr_field)
        attr_field = field_id.intersection(vector.columns).pop()
        # Create a mapping dictionary for continuous or categorical values
        cont_vals_dict = {src: (dst + 1 if continuous else src) for dst, src in enumerate(attr_values)}
        if all(isinstance(val, str) for val in vector[attr_field].unique().tolist()):
            cont_vals_dict = {str(src): dst for src, dst in cont_vals_dict.items()}

        # Add a 'burn_val' column with mapped values
        vector['burn_val'] = vector[attr_field].map(cont_vals_dict)
        burn_field = 'burn_val'
    else:
        # No filtering; use a constant burn value
        burn_field = None

    # Prepare shapes for rasterization
    if burn_field:
        shapes = ((geom, value) for geom, value in zip(vector.geometry, vector[burn_field]))
    else:
        shapes = ((geom, default_burn_value) for geom in vector.geometry)

    # Rasterize the shapes
    label_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        merge_alg=rasterio.enums.MergeAlg.replace,
        dtype=dtype,
    )
    meta = image.meta.copy()
    meta.update(driver='GTiff')
    meta.update(dtype=dtype)
    meta.update(count=1)
    meta.update(nodata=0)
    meta.update(compress='lzw')
    meta.update(blockxsize=256)
    meta.update(blockysize=256)
    meta.update(tiled=True)
    meta.update(BIGTIFF='YES')
    
    if write_raster:
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(label_raster, 1)
        return output_path
    else:
        memfile = MemoryFile()
        with memfile.open(**meta) as dst:
            dst.write(label_raster, 1)
        return memfile.open()