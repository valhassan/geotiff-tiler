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
