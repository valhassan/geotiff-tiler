import gc
import time
import math
import zarr
import tracemalloc
import logging
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from rasterio.windows import Window
from typing import Tuple, Dict, List, Any
from pathlib import Path

from geotiff_tiler.utils.io import load_image, load_mask
from geotiff_tiler.utils.geoutils import create_nodata_mask, apply_nodata_mask, rasterize_vector, get_intersection
from geotiff_tiler.utils.geoutils import ensure_crs_match, clip_to_intersection
from geotiff_tiler.utils.checks import check_image_validity, check_label_validity, calculate_overlap, ResourceManager
from geotiff_tiler.utils.checks import is_image_georeferenced, is_label_georeferenced, check_alignment
from geotiff_tiler.utils.visualization import visualize_zarr_patches
from geotiff_tiler.config.logging_config import logger

logger = logging.getLogger(__name__)


class Tiler:
    """A class for tiling geospatial images and their corresponding labels into patches.

    This class handles the process of splitting geospatial images and their corresponding
    labels (either raster or vector) into smaller patches for machine learning applications.
    It supports various filtering options, handles coordinate reference system (CRS) 
    mismatches, and ensures proper alignment between images and labels.

    Attributes:
        input_dict (List[Dict[str, Any]]): List of dictionaries with image, label, and metadata.
        patch_size (Tuple[int, int]): Size of patches to create (height, width).
        stride (int): Step size between patches. If None, uses max(patch_size).
        discard_empty (bool): Whether to discard patches with no label data.
        label_threshold (float): Minimum ratio of non-zero pixels required in a label patch.
        output_dir (str): Directory where output patches will be saved.
        attr_field (str or List[str]): Field(s) in vector data containing classification attributes.
        attr_values (List[str]): Values in attr_field to use for classification.
    """
    
    
    def __init__(self, 
                 input_dict: List[Dict[str, Any]], 
                 patch_size: Tuple[int, int], # (height, width)
                 stride: int = None,
                 attr_field: str = None,
                 attr_values: List[str] = None,
                 discard_empty: bool = True,
                 label_threshold: float = 0.01, # minimum of non-zero pixels in a patch to be considered valid (0-1)
                 output_dir: str = None,
                 zarr_filename: str = None):
        """Initialize the Tiler with configuration parameters.

        Args:
            input_dict (List[Dict[str, Any]]): List of dictionaries containing:
                - 'image': Path to the image file (str)
                - 'label': Path to the label file (str)
                - 'metadata': Dictionary with additional metadata (Dict)
            patch_size (Tuple[int, int]): Size of patches to create as (height, width).
            stride (int, optional): Step size between patches. If None, uses max(patch_size).
            attr_field (str or List[str], optional): Field(s) in vector data containing classification attributes.
            attr_values (List[str], optional): Values in attr_field to use for classification.
            discard_empty (bool, optional): Whether to discard patches with no label data. Defaults to True.
            label_threshold (float, optional): Minimum ratio of non-zero pixels required in a label patch (0-1). 
                Defaults to 0.01.
            output_dir (str, optional): Directory where output patches will be saved.
        """
        self.input_dict = input_dict
        self.patch_size = patch_size
        self.stride = stride if stride is not None else max(patch_size)
        self.discard_empty = discard_empty
        self.label_threshold = label_threshold
        self.output_dir = output_dir
        self.zarr_filename = zarr_filename
        self.attr_field = attr_field
        self.attr_values = attr_values
        
    @staticmethod
    def pad_patch(patch: np.ndarray, patch_size: Tuple[int, int], mode='edge'):
        """Pads the patch to the patch size"""
        height, width = patch.shape[-2:]
        pad_height = patch_size[0] - height
        pad_width = patch_size[1] - width
        if patch.ndim == 2:
            padded_patch = np.pad(patch, ((0, pad_height), (0, pad_width)), mode=mode)
            return padded_patch
        elif patch.ndim == 3:
            padded_patch = np.pad(patch, ((0, 0), (0, pad_height), (0, pad_width)), mode=mode)
            return padded_patch
        else:
            raise ValueError(f"Invalid patch shape: {patch.shape}")
    
    def _filter_patches(self, label: np.ndarray) -> bool:
        """
        Filters patches based on the discard_empty flag and label_threshold.

        Args:
            label (np.ndarray): The corresponding label patch.

        Returns:
            bool: True if the patch passes the filters, False otherwise.
        """
        # Ensure label is not empty to avoid issues
        if label.size == 0:
            logger.debug("Patch discarded: invalid shape or empty")
            return False
        nonzero_count = np.count_nonzero(label)
        
        # Discard patches where all label values are 0
        if self.discard_empty and nonzero_count == 0:
            logger.debug("Patch discarded: all label values are 0")
            return False
        
        # Apply label coverage threshold
        if self.label_threshold is not None:
            label_coverage = nonzero_count / label.size
            if label_coverage < self.label_threshold:
                logger.debug(f"Patch discarded: label coverage {label_coverage:.2f} < {self.label_threshold}")
                return False

        return True

    def _prepare_vector_labels(self, label: gpd.GeoDataFrame, image: rasterio.DatasetReader):
        """Prepares vector labels for tiling"""
        nodata_mask = create_nodata_mask(image)
        label = apply_nodata_mask(label, nodata_mask)
        label = rasterize_vector(label, image,
                                  attr_field=self.attr_field,
                                  attr_values=self.attr_values,
                                  write_raster=False, 
                                  output_path=None)
        return label
    
    def tiling(self, 
               image: rasterio.DatasetReader, 
               label: rasterio.DatasetReader,
               metadata: Dict[str, Any],
               image_name: str,
               output_dir: str,
               zarr_filename: str):
        """Tile an image and its corresponding label into patches.

        This method divides the input image and label into patches of the specified size using
        the configured stride. It filters patches based on label content and pads patches
        at the image boundaries to match the patch size.

        Args:
            image (rasterio.DatasetReader): The input image to be tiled.
            label (rasterio.DatasetReader): The corresponding label to be tiled.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]: A tuple containing:
                - image_patches: List of image patches as numpy arrays
                - label_patches: List of label patches as numpy arrays
                - positions: List of (x, y) coordinates for each patch in the original image

        Raises:
            AssertionError: If image and label dimensions don't match or patch size exceeds image dimensions.
            Exception: If the tiling process fails for any reason.
        """
        chunk_size = 32
        buffer_size = chunk_size ** 2
        
        image_width = image.width
        image_height = image.height
        label_width = label.width
        label_height = label.height
        
        assert (image_width == label_width and 
                image_height == label_height), "Image and label dimensions must match"
        assert (self.patch_size[0] <= image_height and 
                self.patch_size[1] <= image_width), "Patch size must be smaller than image dimensions"
        
        logger.info(f"Tiling {image_height} x {image_width} "
                    f"image with patch size {self.patch_size} and stride {self.stride}")
        
        number_of_patches_x = math.ceil(image_width / self.stride)
        number_of_patches_y = math.ceil(image_height / self.stride)
        total_patches = number_of_patches_x * number_of_patches_y
        
        
        output_dir_viz = Path(output_dir) / "viz"
        output_dir_viz.mkdir(parents=True, exist_ok=True)
        output_dir_parent = output_dir_viz.parent
        
        if zarr_filename is None:
            zarr_filename = "zarr_file.zarr"
        zarr_path_file = output_dir_parent / zarr_filename
        zarr_path_with_image_name = zarr_path_file / image_name
        
        store = zarr.storage.LocalStore(str(zarr_path_file))
        root = zarr.group(store=store, overwrite=False)
        
        if image_name in root:
            logger.info(f"Image {image_name} already exists in {zarr_path_file}, skipping")
            return None
        image_group = root.create_group(name=image_name, overwrite=False)
        # n_patches_per_chunk = total_patches // max(1, total_patches // 100)
        # print(f"n_patches_per_chunk: {n_patches_per_chunk}")
        
        image_patches_shape = (0, image.count, *self.patch_size)
        label_patches_shape = (0, label.count, *self.patch_size)
        image_patches_chunk_size = (chunk_size, image.count, *self.patch_size)
        label_patches_chunk_size = (chunk_size, label.count, *self.patch_size)
        compressors = [zarr.codecs.BloscCodec(cname='lz4', clevel=5, shuffle='shuffle')]
        
        images_array = image_group.create_array(name='images',
                                                shape=image_patches_shape,
                                                chunks=image_patches_chunk_size,
                                                compressors=compressors,
                                                dtype=image.dtypes[0])
        labels_array = image_group.create_array(name='labels',
                                                shape=label_patches_shape,
                                                chunks=label_patches_chunk_size,
                                                compressors=compressors,
                                                dtype=label.dtypes[0])
        locations_array = image_group.create_array(name='locations',
                                                   shape=(0, 2),
                                                   chunks=(total_patches, 2),
                                                   compressors=compressors,
                                                   dtype=np.int32)
        
        image_buffer = []
        label_buffer = []
        location_buffer = []
        discarded_count = 0
        patch_count = 0
        start_time = time.time()
        try:
            with tqdm(total=total_patches, desc="Tiling patches") as pbar:
                for y in range(0, image_height, self.stride):
                    for x in range(0, image_width, self.stride):
                        window_width = min(self.patch_size[1], image_width - x)
                        window_height = min(self.patch_size[0], image_height - y)
                        window = Window(col_off=x, row_off=y, width=window_width, height=window_height)
                        
                        label_patch = label.read(window=window, boundless=False)
                        if not self._filter_patches(label_patch):
                            discarded_count += 1
                            continue
                        image_patch = image.read(window=window, boundless=False)
                        
                        if image_patch.shape[1:] != self.patch_size or label_patch.shape[1:] != self.patch_size:
                            image_patch = self.pad_patch(image_patch, self.patch_size)
                            label_patch = self.pad_patch(label_patch, self.patch_size)
                        
                        image_buffer.append(image_patch)
                        label_buffer.append(label_patch)
                        location_buffer.append((x, y))
                        
                        if len(image_buffer) >= buffer_size:
                            images_array.append(np.stack(image_buffer, axis=0))
                            labels_array.append(np.stack(label_buffer, axis=0))
                            locations_array.append(np.array(location_buffer))
                            patch_count += len(image_buffer)
                            image_buffer, label_buffer, location_buffer = [], [], []
                        
                        pbar.update(1)
                        
            if len(image_buffer) > 0:
                images_array.append(np.stack(image_buffer, axis=0))
                labels_array.append(np.stack(label_buffer, axis=0))
                locations_array.append(np.array(location_buffer))
                patch_count += len(image_buffer)
                image_buffer, label_buffer, location_buffer = [], [], []
                
            metadata["image_channels"] = image.count
            metadata["label_channels"] = label.count
            image_group.attrs.update(metadata)
            
            # Save visualization
            visualize_zarr_patches(zarr_path=zarr_path_file, 
                                   image_name=image_name, 
                                   number_of_patches=patch_count,
                                   save_path=output_dir_viz / f"{image_name}.png")
            
            logger.info(f"Tiling complete:\n\
                        Extracted {patch_count} patches,\n\
                        Discarded {discarded_count} of {total_patches} patches\n\
                        Saved to {zarr_path_with_image_name}")
            return zarr_path_with_image_name
        except Exception as e:
            logger.error(f"Tiling failed: {e}")
            raise
        finally:
            end_time = time.time()
            logger.info(f"Tiling time: {end_time - start_time:.2f} seconds")
    
    def create_tiles(self):
        """Process all input image-label pairs and create tiles.

        This method iterates through all input image-label pairs, performs validation checks,
        handles CRS mismatches, and generates patches. The patches are saved in zarr format
        and visualizations are created. The process includes:
        
        1. Loading images and labels
        2. Validating georeference information
        3. Checking image and label validity
        4. Ensuring CRS match between image and label
        5. Calculating overlap and clipping to intersection
        6. Converting vector labels to raster if needed
        7. Creating and saving tiles
        
        Returns:
            None: Results are saved to the specified output_dir.
            
        Raises:
            Exception: If the tiling process fails for any reason.
        """
        tracemalloc.start()
        resource_manager = ResourceManager()
        
        try:
            zarr_paths = []
            for id, input_dict in tqdm(enumerate(self.input_dict), 
                                    total=len(self.input_dict), desc="Processing input pairs"):
                image_path = input_dict["image"]
                label_path = input_dict["label"]
                metadata = input_dict["metadata"]
                image_name = Path(image_path).stem
                
                metadata["image_name"] = image_name
                metadata["patch_size"] = self.patch_size
                metadata["stride"] = self.stride
                
                try:
                    image = resource_manager.register(load_image(image_path))
                    label = resource_manager.register(load_mask(label_path))
                except Exception as e:
                    logger.error(f"Error loading image or label: {e}")
                    resource_manager.close_all()
                    continue
                
                if isinstance(label, gpd.GeoDataFrame) and (not is_image_georeferenced(image) or 
                        not is_label_georeferenced(label)):
                    logger.info("Skipping pair due to invalid label or image")
                    continue
                
                if isinstance(label, rasterio.DatasetReader) and (not is_image_georeferenced(image) or 
                                                                not is_label_georeferenced(label)):
                    if check_alignment(image, label):
                        try:
                            zarr_path = self.tiling(image, label, metadata, 
                                                    image_name, self.output_dir, self.zarr_filename)
                            if zarr_path is None:
                                continue
                            zarr_paths.append(zarr_path)
                        except Exception as e:
                            logger.error(f"Error processing image {image_name}: {e}")
                        finally:
                            resource_manager.close_all()
                        continue
                    else:
                        logger.info("Skipping pair due to invalid label or image")
                        resource_manager.close_all()
                        continue
                
                image_valid, image_msg = check_image_validity(image)
                if not image_valid:
                    logger.info(f"Skipping pair due to invalid image: {image_msg}")
                    continue
                
                label_valid, label_msg = check_label_validity(label)
                if not label_valid:
                    logger.info(f"Skipping pair due to invalid label: {label_msg}")
                    continue
                
                image, label, was_converted = ensure_crs_match(image, label)
                if was_converted:
                    logger.info("CRS mismatch between image and label, reprojecting label to match image")
                    if isinstance(label, rasterio.DatasetReader):
                        resource_manager.register(label)
                
                overlap_pct, overlap_msg = calculate_overlap(image, label)
                if overlap_pct == 0:
                    logger.info(f"Skipping pair due to no overlap: {overlap_msg}")
                    continue
                logger.info(f"Processing pair with {overlap_msg}")
                
                intersection = get_intersection(image, label)
                if intersection is None:
                    logger.info("Skipping pair: No intersection between image and label")
                    continue
                image, label = clip_to_intersection(image, label, intersection)
                resource_manager.register(image)
                resource_manager.register(label)
                
                if isinstance(label, gpd.GeoDataFrame):                    
                    label = self._prepare_vector_labels(label, image)
                    resource_manager.register(label)
                
                zarr_path = self.tiling(image, label, metadata, 
                                        image_name, self.output_dir, self.zarr_filename)
                if zarr_path is None:
                    continue
                zarr_paths.append(zarr_path)
                current, peak = tracemalloc.get_traced_memory()
                resource_manager.close_all()
                logger.info(f"[Item {id}] Memory before processing: {current/1024/1024:.2f}MB, Peak: {peak/1024/1024:.2f}MB")
            if zarr_paths:
                df = pd.DataFrame(zarr_paths)
                csv_file_name = "zarr_paths.csv" if self.zarr_filename is None else f"{self.zarr_filename}.csv"
                csv_path = Path(self.output_dir) / csv_file_name
                df.to_csv(csv_path, index=False, header=False, mode='a')
            logger.info("Tiling complete")
        except Exception as e:
            logger.error(f"Tiling process failed: {e}")
            raise
        finally:
            resource_manager.close_all()
            tracemalloc.stop()



if __name__ == '__main__':
    
    data = [{"image": "https://int.datacube.services.geo.ca/stac/api/collections/worldview-2-ortho-pansharp/items/ON_Gore-Bay_WV02_20110828", 
             "label": "/home/valhassa/Projects/geotiff-tiler/data/ON45.gpkg",
             "metadata": {"collection": "worldview-2-ortho-pansharp", "gsd": 0.46, "lat_lon":(45.0, -75.0), "datetime":"2016-10-02T18:40:15Z"}}, 
            {"image": "/home/valhassa/Projects/geotiff-tiler/data/AB26_NRGB_8bit_clahe25.tif", 
             "label": "/home/valhassa/Projects/geotiff-tiler/data/AB26.gpkg",
             "metadata": {"collection": "planetscope", "gsd": 0.6, "lat_lon":(45.0, -75.0), "datetime":"2020-01-01T00:00:00Z"}}, 
            {"image": "/home/valhassa/Projects/geotiff-tiler/data/GF2_PMS1__L1A0000564539-MSS1.tif", 
             "label": "/home/valhassa/Projects/geotiff-tiler/data/GF2_PMS1__L1A0000564539-MSS1_24label.tif",
             "metadata": {"collection": "gaofen-2-pansharp", "gsd": 0.8, "lat_lon":(45.0, -75.0), "datetime":"2020-01-01T00:00:00Z"}}]
    
    tiler = Tiler(input_dict=data, 
                  patch_size=(1024, 1024),
                  attr_field=["class", "Quatreclasses"],
                  attr_values=[1, 2, 3, 4],
                  stride=1024, 
                  discard_empty=True, 
                  label_threshold=0.1,
                  output_dir='/home/valhassa/Projects/geotiff-tiler/data/output')
    
    tiler.create_tiles()
    