import os
import gc
import time
import json
import math
import zarr
import tracemalloc
import logging
import rasterio
import numpy as np
import pandas as pd
import webdataset as wds
import geopandas as gpd
from tqdm import tqdm
from rasterio.windows import Window
from collections import defaultdict
from typing import Tuple, Dict, List, Any
from pathlib import Path
from datetime import datetime

from geotiff_tiler.utils.io import load_image, load_mask
from geotiff_tiler.utils.geoutils import create_nodata_mask, apply_nodata_mask, rasterize_vector, get_intersection
from geotiff_tiler.utils.geoutils import ensure_crs_match, clip_to_intersection
from geotiff_tiler.utils.checks import validate_pair, calculate_overlap, ResourceManager
from geotiff_tiler.utils.visualization import visualize_zarr_patches
from geotiff_tiler.config.logging_config import logger
from geotiff_tiler.val import calculate_class_distribution, create_spatial_grid, select_validation_cells
from geotiff_tiler.tiling_manifest import TilingManifest

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
                 bands_requested: List[str] = ["red", "green", "blue", "nir"],
                 band_indices: List[int] = None,
                 stride: int = None,
                 grid_size: int = 4,
                 val_ratio: float = 0.2,
                 class_balance_weight: float = 0.5,
                 spatial_weight: float = 0.5,
                 attr_field: str = None,
                 attr_values: List[str] = None,
                 class_ids: Dict[str, int] = None,
                 discard_empty: bool = True,
                 label_threshold: float = 0.01, # minimum of non-zero pixels in a patch to be considered valid (0-1)
                 split: str = "trn",
                 prefix: str = "sample",
                 output_dir: str = None
                 ):
        """Initialize the Tiler with configuration parameters.

        Args:
            input_dict (List[Dict[str, Any]]): List of dictionaries containing:
                - 'image': Path to the image file (str)
                - 'label': Path to the label file (str)
                - 'metadata': Dictionary with additional metadata (Dict)
            patch_size (Tuple[int, int]): Size of patches to create as (height, width).
            stride (int, optional): Step size between patches. If None, uses max(patch_size).
            grid_size (int, optional): Size of the grid to create. Defaults to 4.
            val_ratio (float, optional): Ratio of validation cells to total cells. Defaults to 0.2.
            class_balance_weight (float, optional): Weight for class balance. Defaults to 0.5.
            spatial_weight (float, optional): Weight for spatial balance. Defaults to 0.5.
            attr_field (str or List[str], optional): Field(s) in vector data containing classification attributes.
            attr_values (List[str], optional): Values in attr_field to use for classification.
            class_ids (Dict[str, int], optional): Dictionary mapping class names to class IDs.
            discard_empty (bool, optional): Whether to discard patches with no label data. Defaults to True.
            label_threshold (float, optional): Minimum ratio of non-zero pixels required in a label patch (0-1). 
                Defaults to 0.01.
            split (str, optional): Split to use for tiling. Defaults to "trn".
            prefix (str, optional): Prefix for output files. Defaults to "sample".
            output_dir (str, optional): Directory where output patches will be saved.
        """
        self.input_dict = input_dict
        self.patch_size = patch_size
        self.bands_requested = bands_requested
        self.band_indices = band_indices
        self.stride = stride if stride is not None else max(patch_size)
        self.discard_empty = discard_empty
        self.label_threshold = label_threshold
        self.split = split if split in ["trn", "tst"] else "trn"
        self.output_dir = output_dir
        self.prefix = prefix
        self.attr_field = attr_field
        self.attr_values = attr_values
        self.grid_size = grid_size
        self.val_ratio = val_ratio
        self.class_balance_weight = class_balance_weight
        self.spatial_weight = spatial_weight
        self.class_ids = class_ids or {'background': 0,
                                       'fore': 1,
                                       'hydro': 2,
                                       'road': 3,
                                       'building': 4
                                       }
        
        self.manifest = TilingManifest(output_dir, self.prefix)
    
    def create_tiles(self):
        """Process all input image-label pairs and create tiles.

        This method iterates through all input image-label pairs, performs validation checks,
        handles CRS mismatches, and generates patches. Results are saved to the specified output_dir.
        
        Returns:
            dict: Summary of processing results including successes, failures, and skips.
        """
        image_analyses = []
        global_class_distribution = defaultdict(list)
        processing_summary = {"total": len(self.input_dict), "successful": 0, "skipped": 0, "failed": 0}
        
        tracemalloc.start()
        resource_manager = ResourceManager()
        
        logger.info("Phase 1: Analyzing images")
        for input_dict in tqdm(self.input_dict, desc="Processing input pairs"):
            image_path = input_dict["image"]
            label_path = input_dict["label"]
            metadata = input_dict["metadata"]
            image_name = Path(image_path).stem
            
            # Skip already completed images
            if self.manifest.is_image_completed(image_name):
                logger.info(f"Skipping already completed image: {image_name} ")
                processing_summary["skipped"] += 1
                continue
            
            # Mark image as in progress
            self.manifest.mark_image_in_progress(image_name)
            
            metadata["image_name"] = image_name
            metadata["patch_size"] = self.patch_size
            metadata["stride"] = self.stride
            sensor_type = metadata.get("collection", "unknown")
            
            result = self._process_single_pair(image_path, label_path, resource_manager)
            processing_summary[result["status"]] += 1
            if result["status"] != "successful":
                logger.info(f"Pair {input_dict['image']} - {result['reason']}")
                self.manifest.mark_image_failed(image_name, result["reason"])
                processing_summary["failed"] += 1
                resource_manager.close_all()
                continue
            self._process_analysis(result["image"], result["label"], image_path, label_path, metadata,
                                   image_name, sensor_type, image_analyses, global_class_distribution)
            self.manifest.update_class_distribution(image_analyses[-1]["class_distribution"])
            resource_manager.close_all()
            current, peak = tracemalloc.get_traced_memory()
            logger.info(f"Current memory: {current/1024/1024:.2f}MB, Peak memory: {peak/1024/1024:.2f}MB")
        target_distribution = {cls: np.mean(values) for cls, values in global_class_distribution.items()}
        
        logger.info(f"Phase 2: Creating WebDataset files with pre-determined splits")
    
        self.prefix_shard_indices = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
        self.prefix_shard_sizes = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
        self.prefix_patch_counts = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
        
        for split in ["trn", "val", "tst"]:
            index, size, count = self.manifest.get_shard_info(self.prefix, split)
            self.prefix_shard_indices[self.prefix][split] = index
            self.prefix_shard_sizes[self.prefix][split] = size
            self.prefix_patch_counts[self.prefix][split] = count
        
        self.prefix_writers = {}
        
        create_val_set = False
        for analysis in tqdm(image_analyses, desc="Creating WebDataset files"):
            image_name = analysis['image_name']
            
            if self.manifest.is_image_completed(image_name):
                logger.info(f"Skipping already completed image for tiling: {image_name} ")
                processing_summary["skipped"] += 1
                continue
            
            if self.split == "trn":
                val_ratio = self.manifest.get_validation_ratio(self.val_ratio)
                validation_cells = select_validation_cells(analysis['grid'],
                                                           target_distribution, 
                                                           val_ratio,
                                                           self.class_balance_weight,
                                                           self.spatial_weight
                                                           )
                create_val_set = True
            else:
                validation_cells = None
            
            self.manifest.mark_image_in_progress(image_name)
            try:
                image_path = analysis['image_path']
                label_path = analysis['label_path']
                
                results = self._process_single_pair(image_path, label_path, resource_manager)
                if results['status'] != 'successful':
                    logger.info(f"Pair {image_path} - {results['reason']}")
                    self.manifest.mark_image_failed(image_name, results['reason'])
                    processing_summary["failed"] += 1
                    resource_manager.close_all()
                    continue
                
                image = results['image']
                label = results['label']
                analysis['image'] = image
                analysis['label'] = label
                
                self._tiling(analysis, validation_cells, create_val_set)
                self.manifest.mark_image_completed(image_name)
                resource_manager.close_all()
            except Exception as e:
                logger.error(f"Error tiling image {image_name}: {e}")
                self.manifest.mark_image_failed(image_name, str(e))
                processing_summary["failed"] += 1
                resource_manager.close_all()
            
        for prefix, writers in self.prefix_writers.items():
            for split, writer in writers.items():
                writer.close()
        self.manifest.save_manifest()
        logger.info(f"Final checkpoint saved")            
        
        logger.info(f"\nProcessing complete. Summary: {processing_summary}")
        total_sizes = self.manifest.get_total_sizes_by_split()
        for prefix, counts in self.prefix_patch_counts.items():
            def get_shard_count(split_name):
                if counts[split_name] > 0:
                    return self.prefix_shard_indices[prefix][split_name] + 1
                else:
                    return 0
            
            trn_shards = get_shard_count('trn')
            val_shards = get_shard_count('val')
            tst_shards = get_shard_count('tst')
                
            logger.info(f"""
                        Prefix: {prefix}, 
                        Training patches: {counts['trn']}, 
                        Validation patches: {counts['val']}, 
                        Test patches: {counts['tst']},
                        Total patches: {sum(counts.values())},
                        Training size: {total_sizes['trn'] / 1024**2:.2f} MB,
                        Validation size: {total_sizes['val'] / 1024**2:.2f} MB,
                        Test size: {total_sizes['tst'] / 1024**2:.2f} MB,
                        Training shards: {trn_shards},
                        Validation shards: {val_shards},
                        Test shards: {tst_shards},
                        """)
        self.export_normalization_stats()
        
        return processing_summary
    
    def export_normalization_stats(self, output_path: str = None):
        """
        Export normalization statistics to a JSON file.
        
        Args:
            output_path (str, optional): Path to save statistics. 
                                    If None, saves to output_dir/normalization_stats.json
        """
        if output_path is None:
            output_path = Path(self.output_dir) / "normalization_stats.json"
        
        try:
            all_stats = self.manifest.get_all_dataset_statistics()
            
            stats_with_metadata = {
                "created_at": datetime.now().isoformat(),
                "dataset_prefix": self.prefix,
                "statistics": all_stats
            }
            
            with open(output_path, 'w') as f:
                json.dump(stats_with_metadata, f, indent=2)
            
            logger.info(f"Normalization statistics saved to {output_path}")
            
            for prefix, stats in all_stats.items():
                logger.info(f"\nStatistics for {prefix}:")
                logger.info(f"  Bands: {stats['band_count']}")
                logger.info(f"  Patches processed: {stats['patch_count']}")
                logger.info(f"  Mean: {[f'{m:.4f}' for m in stats['mean']]}")
                logger.info(f"  Std:  {[f'{s:.4f}' for s in stats['std']]}")
                
        except Exception as e:
            logger.error(f"Failed to export normalization statistics: {e}")
    
    def retry_failed_images(self, max_retries: int = 3) -> Dict[str, Any]:
        """Retry all failed images from the manifest.
        
        Args:
            max_retries (int): Maximum number of retry attempts for each failed image.
                            Defaults to 3.
        
        Returns:
            dict: Summary of retry processing results including successes, failures, and skips.
        """
        # Get current failed images from manifest
        failed_images = self.manifest.failed_images.copy()
        
        if not failed_images:
            logger.info("No failed images to retry")
            return {"total": 0, "successful": 0, "skipped": 0, "failed": 0}
        
        logger.info(f"Found {len(failed_images)} failed images to retry")
        
        # Find the failed images in the original input_dict
        retry_dict = []
        for input_item in self.input_dict:
            image_name = Path(input_item["image"]).stem
            if image_name in failed_images:
                retry_dict.append(input_item)
                logger.info(f"Queuing {image_name} for retry (previous failure: {failed_images[image_name]})")
        
        if not retry_dict:
            logger.warning("Failed images not found in input_dict. They may have been removed.")
            return {"total": len(failed_images), "successful": 0, "skipped": 0, "failed": len(failed_images)}
        
        # Store original input_dict and temporarily replace with retry list
        original_input_dict = self.input_dict
        self.input_dict = retry_dict
        
        # Track retry attempts
        retry_summary = {"total": len(retry_dict), "successful": 0, "skipped": 0, "failed": 0}
        
        # Perform retries with attempt tracking
        for attempt in range(1, max_retries + 1):
            if not self.input_dict:  # All images processed successfully
                break
                
            logger.info(f"\n=== Retry attempt {attempt}/{max_retries} ===")
            
            # Process the failed images
            result = self.create_tiles()
            
            # Update retry summary
            retry_summary["successful"] += result["successful"]
            retry_summary["skipped"] += result["skipped"]
            
            # Check which images are still failing
            still_failed = []
            for input_item in self.input_dict:
                image_name = Path(input_item["image"]).stem
                if self.manifest.is_image_failed(image_name):
                    still_failed.append(input_item)
            
            # Update input_dict with only still-failed images for next attempt
            self.input_dict = still_failed
            
            if not still_failed:
                logger.info(f"All images processed successfully after {attempt} attempt(s)")
                break
            
            logger.info(f"{len(still_failed)} images still failing after attempt {attempt}")
            
            if attempt < max_retries:
                # Optional: Add a small delay between retries to avoid hammering the server
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                logger.info(f"Waiting {wait_time} seconds before next retry...")
                time.sleep(wait_time)
        
        # Final tally of failed images
        retry_summary["failed"] = len(self.input_dict)
        
        # Restore original input_dict
        self.input_dict = original_input_dict
        
        # Log final summary
        logger.info(f"\n=== Retry Summary ===")
        logger.info(f"Total images retried: {retry_summary['total']}")
        logger.info(f"Successfully processed: {retry_summary['successful']}")
        logger.info(f"Skipped: {retry_summary['skipped']}")
        logger.info(f"Still failing: {retry_summary['failed']}")
        
        if retry_summary["failed"] > 0:
            logger.warning("The following images are still failing after all retry attempts:")
            for input_item in self.input_dict:
                image_name = Path(input_item["image"]).stem
                reason = self.manifest.failed_images.get(image_name, "Unknown reason")
                logger.warning(f"  - {image_name}: {reason}")
        
        return retry_summary
    
    def _process_single_pair(self, image_path, label_path, resource_manager):
        """Process a single image-label pair and return the result status."""
        image_name = Path(image_path).stem
        try:
            # Load data
            image = resource_manager.register(load_image(image_path, self.bands_requested, self.band_indices))
            label = resource_manager.register(load_mask(label_path))
            
            # Validate pair and determine processing path
            validation_result = validate_pair(image, label)
            if not validation_result["valid"]:
                return {"status": "skipped", "reason": validation_result["reason"]}
                
            special_case = validation_result.get("special_case", False)
            if special_case:
                # Special case processing (e.g., non-georeferenced but aligned)
                return {"image": image, "label": label, "status": "successful", "reason": "Processed as special case"}
            
            # Standard processing: CRS, overlap, intersection
            image, label, was_converted = ensure_crs_match(image, label)
            if was_converted:
                logger.info(f"CRS mismatch for {image_name}, reprojecting label")
                if isinstance(label, rasterio.DatasetReader):
                    resource_manager.register(label)
            
            overlap_pct, overlap_msg = calculate_overlap(image, label)
            if overlap_pct == 0:
                return {"status": "skipped", "reason": f"No overlap: {overlap_msg}"}
            logger.info(f"Processing pair {image_name} with {overlap_msg}")
            
            intersection = get_intersection(image, label)
            if intersection is None:
                return {"status": "skipped", "reason": "No intersection between image and label"}
            image, label = clip_to_intersection(image, label, intersection)
            resource_manager.register(image)
            resource_manager.register(label)
            
            # Convert vector labels if necessary
            if isinstance(label, gpd.GeoDataFrame):
                label = self._prepare_vector_labels(label, image)
                resource_manager.register(label)

            return {"image": image, "label": label, "status": "successful", "reason": "Processed successfully"}
            
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def _process_analysis(self, image, label, image_path, label_path, metadata, image_name, sensor_type, 
                     image_analyses, global_class_distribution):
        """
        Process class distribution and spatial grid for an image-label pair and store results.
        
        Args:
            image: Loaded image data
            label: Loaded label data
            image_path (str): Path to image file
            label_path (str): Path to label file
            metadata (dict): Metadata for the pair
            image_name (str): Name of the image
            sensor_type (str): Type of sensor
            image_analyses (list): List to append analysis results to
            global_class_distribution (defaultdict): Dictionary to store class distribution
            
        Returns:
            None
        """
        class_distribution = calculate_class_distribution(label, self.class_ids)
        grid = create_spatial_grid(image, self.stride, self.grid_size)
        
        image_analyses.append({
            'image_path': image_path,
            'label_path': label_path,
            'metadata': metadata,
            'image_name': image_name,
            'sensor_type': sensor_type,
            'class_distribution': class_distribution,
            'grid': grid
        })
        for cls, value in class_distribution.items():
            global_class_distribution[cls].append(value)
    
    def _tiling(self,
               image_analysis: Dict[str, Any],
               validation_cells: List[str] = None,
               create_val_set: bool = False
               ):
        
        image = image_analysis['image']
        label = image_analysis['label']
        
        
        
        image_width = image.width
        image_height = image.height
        image_bands = image.count
        label_width = label.width
        label_height = label.height
        label_bands = label.count
        
        metadata = image_analysis['metadata']
        metadata["image_channels"] = image_bands
        metadata["label_channels"] = label_bands
        
        total_patches = image_analysis['grid']['total_patches']
        grid_size = image_analysis['grid']['grid_size']
        image_name = image_analysis['image_name']
        
        assert (image_width == label_width and 
                image_height == label_height), "Image and label dimensions must match"
        assert (self.patch_size[0] <= image_height and 
                self.patch_size[1] <= image_width), "Patch size must be smaller than image dimensions"
        self.manifest.update_image_metadata(image_name, {"path": image_analysis['image_path'],
                                                         "label_path": image_analysis['label_path'],
                                                         "metadata": metadata,
                                                         "sensor_type": metadata.get("collection", "unknown"),
                                                         "class_distribution": 
                                                             image_analysis.get('class_distribution', {}),
                                                         })
        if create_val_set:
            output_train_dir = Path(self.output_dir) / self.prefix / "trn"
            output_val_dir = Path(self.output_dir) / self.prefix / "val"
            output_train_dir.mkdir(parents=True, exist_ok=True)
            output_val_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_tst_dir = Path(self.output_dir) / self.prefix / "tst"
            output_tst_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Tiling {image_height} x {image_width} x {image_bands} "
                    f"image with patch size {self.patch_size} and stride {self.stride}")
        
        if not hasattr(self, 'prefix_shard_indices'):
            self.prefix_shard_indices = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
            self.prefix_shard_sizes = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
            self.prefix_writers = {}
            self.prefix_patch_counts = defaultdict(lambda: {"trn": 0, "val": 0, "tst": 0})
        
        if self.prefix not in self.prefix_writers and create_val_set:
            trn_shard_path = self._get_shard_path(output_train_dir, self.prefix, "trn",
                                                  self.prefix_shard_indices[self.prefix]["trn"])
            val_shard_path = self._get_shard_path(output_val_dir, self.prefix, "val", 
                                                  self.prefix_shard_indices[self.prefix]["val"])
            self.prefix_writers[self.prefix] = {"trn": wds.TarWriter(trn_shard_path),
                                               "val": wds.TarWriter(val_shard_path)}
        elif self.prefix not in self.prefix_writers and not create_val_set:
            tst_shard_path = self._get_shard_path(output_tst_dir, self.prefix, "tst", 
                                                  self.prefix_shard_indices[self.prefix]["tst"])
            self.prefix_writers[self.prefix] = {"tst": wds.TarWriter(tst_shard_path)}
        
        MAX_SHARD_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
        discarded_count = 0
        patch_count = 0
        start_time = time.time()
        try:
            with tqdm(total=total_patches, desc="Tiling patches") as pbar:
                
                for y in range(0, image_height, self.stride):
                    for x in range(0, image_width, self.stride):
                        
                        if self.manifest.is_patch_completed(image_name, x, y):
                            patch_count += 1
                            pbar.update(1)
                            continue
                        
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
                        
                        # Determine grid cell for this patch
                        if create_val_set:
                            grid_x = int(x // (image.width / grid_size))
                            grid_y = int(y // (image.height / grid_size))
                            cell_id = f"{grid_x}_{grid_y}"
                            if cell_id in validation_cells:
                                split = "val"
                            else:
                                split = "trn"
                        else:
                            split = "tst"
                        
                        patch_key = f"{self.prefix}_{image_name}_{x}_{y}"
                        
                        all_metadata = {"patch_metadata": {"patch_id": patch_key,
                                                           "pixel_coordinates": [x, y],
                                                           "patch_size": self.patch_size,
                                                           "stride": self.stride,
                                                           "split": split,
                                                           "image_dtype": image.dtypes[0],
                                                           "label_dtype": label.dtypes[0],
                                                           "image_name": image_name,
                                                           "sensor_type": metadata.get("collection", "unknown")
                                                           },
                                        "metadata": metadata
                                        }
                        if split == "trn":
                            try:
                                self.manifest.update_running_statistics(self.prefix, image_patch)
                            except Exception as e:
                                logger.error(f"Error updating running statistics: {e}")
                        patch_size_bytes = self._estimate_patch_size(image_patch, label_patch, all_metadata)
                        current_shard_size = self.prefix_shard_sizes[self.prefix][split]
                        if current_shard_size + patch_size_bytes > MAX_SHARD_SIZE_BYTES:
                            self.prefix_writers[self.prefix][split].close()
                            self.manifest.close_shard(self.prefix, split, 
                                                      self.prefix_shard_indices[self.prefix][split])
                            self.prefix_shard_indices[self.prefix][split] += 1
                            if create_val_set:
                                if split == "trn":
                                    new_shard_path = self._get_shard_path(output_train_dir, self.prefix, "trn",
                                                                          self.prefix_shard_indices[self.prefix]["trn"])
                                else:
                                    new_shard_path = self._get_shard_path(output_val_dir, self.prefix, "val",
                                                                          self.prefix_shard_indices[self.prefix]["val"])
                            else:
                                new_shard_path = self._get_shard_path(output_tst_dir, self.prefix, "tst",
                                                                      self.prefix_shard_indices[self.prefix]["tst"])
                            self.prefix_writers[self.prefix][split] = wds.TarWriter(new_shard_path)
                            self.manifest.update_shard_record(self.prefix, split, 
                                                              self.prefix_shard_indices[self.prefix][split], 
                                                              0, 0, "OPEN", [image_name])
                            logger.info(f"Created new {split} shard: {new_shard_path}")
                            self.prefix_shard_sizes[self.prefix][split] = 0
                            
                        self.prefix_writers[self.prefix][split].write({"__key__": patch_key,
                                                                      "image_patch.npy": image_patch,
                                                                      "label_patch.npy": label_patch,
                                                                      "metadata.json": all_metadata})
                        self.prefix_shard_sizes[self.prefix][split] += patch_size_bytes
                        self.manifest.mark_patch_completed(image_name, x, y)
                        self.manifest.update_shard_info(self.prefix, split, 
                                                          self.prefix_shard_indices[self.prefix][split],
                                                          self.prefix_shard_sizes[self.prefix][split],
                                                          self.prefix_patch_counts[self.prefix][split])
                        self.manifest.update_image_patch_info(image_name, split, 
                                                              self.prefix_shard_indices[self.prefix][split])
                        if patch_count % 100 == 0:
                            self.manifest.save_manifest()
                        
                        self.prefix_patch_counts[self.prefix][split] += 1
                        patch_count += 1
                        pbar.update(1)
            logger.info(f"""
                        Tiling Complete for {image_name}!
                        Extracted patches: {patch_count}
                        Discarded patches: {discarded_count}
                        Total patches: {total_patches}
                        """)
            self.manifest.save_manifest()
        except Exception as e:
            self.manifest.save_manifest()
            logger.error(f"Tiling failed: {e}")
            raise
        finally:
            end_time = time.time()
            logger.info(f"Tiling time: {end_time - start_time:.2f} seconds")
       
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
    
    def _get_shard_path(self, base_path, prefix, split, idx):
        return os.path.join(base_path, f"{prefix}-{split}-{idx:06d}.tar")
    
    def _estimate_patch_size(self, image_patch, label_patch, metadata):
        size = image_patch.nbytes + label_patch.nbytes
        size += len(json.dumps(metadata).encode("utf-8"))
        return size

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
    
    initial_result = tiler.create_tiles()
    if initial_result["failed"] > 0:
        retry_result = tiler.retry_failed_images(max_retries=3)