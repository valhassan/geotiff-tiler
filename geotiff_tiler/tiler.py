import os
import shutil
import gc
import time
import json
import psutil
import tracemalloc
import logging
import rasterio
import numpy as np
import webdataset as wds
from tqdm import tqdm
from rasterio.windows import Window
from collections import defaultdict
from typing import Tuple, Dict, List, Any
from pathlib import Path
from datetime import datetime

from geotiff_tiler.utils.io import validate_image, validate_mask, validate_pair, log_stage
from geotiff_tiler.utils.io import ensure_crs_match, calculate_overlap, clip_to_intersection
from geotiff_tiler.utils.io import prepare_vector_labels
from geotiff_tiler.utils.visualization import visualize_webdataset_patches, create_dataset_summary_visualization
from geotiff_tiler.config.logging_config import logger
from geotiff_tiler.val import calculate_class_distribution, create_spatial_grid, select_validation_cells
from geotiff_tiler.tiling_manifest import TilingManifest

logger = logging.getLogger(__name__)

def log_memory_usage(stage: str, image_name: str = "", force_gc: bool = True):
    """Log current memory usage with optional garbage collection."""
    if force_gc:
        gc.collect()
    
    # Process memory
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    
    # Tracemalloc if available
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        logger.info(f"Memory at {stage} {image_name}: "
                   f"RSS={memory_mb:.1f}MB, "
                   f"Traced={current_mb:.1f}MB, "
                   f"Peak={peak_mb:.1f}MB")
    else:
        logger.info(f"Memory at {stage} {image_name}: RSS={memory_mb:.1f}MB")
    
    return memory_mb

class Tiler:
    """
    A class for tiling geospatial images and their corresponding labels into patches.
    This class handles the process of splitting geospatial images and their corresponding
    labels (either raster or vector) into smaller patches for machine learning applications.
    It supports various filtering options, handles coordinate reference system (CRS) 
    mismatches, and ensures proper alignment between images and labels.
    """
    
    def __init__(self, 
                 input_dict: List[Dict[str, Any]], 
                 patch_size: Tuple[int, int], # (height, width)
                 bands_requested: List[str] = ["red", "green", "blue", "nir"],
                 band_indices: List[int] = None,
                 stride: int = None,
                 grid_size: int = 16,
                 val_ratio: float = 0.1,
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
        # tracemalloc.start()
        processing_summary = {"total": len(self.input_dict), "successful": 0, "skipped": 0, "failed": 0}
        image_analyses = []
        global_class_distribution = defaultdict(list)
        tmp_dir = Path(self.output_dir) / self.prefix / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Phase 1: Processing images for patch generation")
        # log_memory_usage("Phase 1 Start", force_gc=True)
        for input_dict in tqdm(self.input_dict, desc="Processing input pairs"):
            image_path = input_dict["image"]
            label_path = input_dict["label"]
            metadata = input_dict["metadata"]
            image_name = Path(image_path).stem
            image_tmp_dir = tmp_dir / image_name
            if image_tmp_dir.exists():
                shutil.rmtree(image_tmp_dir)
            else:
                image_tmp_dir.mkdir()
            if self.manifest.is_image_completed(image_name):
                logger.info(f"Skipping already completed image: {image_name} ")
                processing_summary["skipped"] += 1
                continue
            self.manifest.mark_image_in_progress(image_name)
            metadata["image_name"] = image_name
            metadata["patch_size"] = self.patch_size
            metadata["stride"] = self.stride
            sensor_type = metadata.get("collection", "unknown")
            result = self.process_single_pair(image_path, label_path, image_tmp_dir)
            processing_summary[result["status"]] += 1
            if result["status"] != "successful":
                logger.info(f"Pair {input_dict['image']} - {result['reason']}")
                self.manifest.mark_image_failed(image_name, result["reason"])
                processing_summary["failed"] += 1
                continue
            logger.info(f"Processing analysis for {image_name}")
            self.process_analysis(result["image_path"], 
                                    result["label_path"], 
                                    metadata, image_name, sensor_type, image_analyses, global_class_distribution)
            logger.info(f"Updating class distribution for {image_name}")
            self.manifest.update_class_distribution(image_analyses[-1]["class_distribution"])
        target_distribution = {cls: np.mean(values) for cls, values in global_class_distribution.items()}
        # log_memory_usage("Phase 1 End", force_gc=True)
        
        logger.info(f"Phase 2: Creating WebDataset files with pre-determined splits")
        # log_memory_usage("Phase 2 Start", force_gc=True)
    
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
            # memory_before = log_memory_usage(f"Before", image_name, force_gc=True)
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
                self.tiling(analysis['image_path'], analysis['label_path'], analysis, validation_cells, create_val_set)
                self.manifest.mark_image_completed(image_name)
            # memory_after = log_memory_usage(f"After", image_name, force_gc=True)
            # logger.info(f"Memory growth: {memory_after - memory_before:.2f}MB")
            except Exception as e:
                logger.error(f"Error tiling image {image_name}: {e}")
                self.manifest.mark_image_failed(image_name, str(e))
                processing_summary["failed"] += 1
            finally:
                self._close_all_writers(flush_only=True)
                self.manifest.save_manifest()
                
        for prefix in list(self.prefix_writers.keys()):
            for split in list(self.prefix_writers[prefix].keys()):
                self._close_writer(prefix, split, flush_only=False)
        self.manifest.save_manifest()
        self.create_summary_visualization(self.output_dir, self.prefix, samples_per_split=5)            
        logger.info(f"Processing complete. Summary: {processing_summary}")
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
                        Total Stats for prefix: {prefix} \n
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
        result = self.manifest.validate_manifest_consistency()
        counts = result['counts']
        
        if result['is_consistent']:
            logger.info(f"""
                        Manifest validation: PASSED \n
                        Images:{counts['from_images']}, 
                        Shards:{counts['from_shards']}, 
                        Stats:{counts['from_statistics']}, 
                        Running:{counts['from_running_stats']}
                        """)
        else:
            issues_summary = ', '.join(result['issues'])
            logger.info(f"""
                        Manifest validation: FAILED ({len(result['issues'])} issues: {issues_summary}) \n
                        Images:{counts['from_images']}, 
                        Shards:{counts['from_shards']}, 
                        Stats:{counts['from_statistics']}, 
                        Running:{counts['from_running_stats']}
                        """)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return processing_summary
    
    @log_stage(stage_name="export_normalization_stats", log_memory=True)
    def export_normalization_stats(self, output_path: str = None):
        """Export normalization statistics to a JSON file."""
        if output_path is None:
            output_path = Path(self.output_dir) / self.prefix / f"{self.prefix}_normalization_stats.json"
        
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
        """Retry all failed images from the manifest."""
        failed_images = self.manifest.failed_images.copy()
        
        if not failed_images:
            logger.info("No failed images to retry")
            return {"total": 0, "successful": 0, "skipped": 0, "failed": 0}
        
        logger.info(f"Found {len(failed_images)} failed images to retry")
        
        retry_dict = []
        for input_item in self.input_dict:
            image_name = Path(input_item["image"]).stem
            if image_name in failed_images:
                retry_dict.append(input_item)
                logger.info(f"Queuing {image_name} for retry (previous failure: {failed_images[image_name]})")
        
        if not retry_dict:
            logger.warning("Failed images not found in input_dict. They may have been removed.")
            return {"total": len(failed_images), "successful": 0, "skipped": 0, "failed": len(failed_images)}
        
        original_input_dict = self.input_dict
        self.input_dict = retry_dict
        
        retry_summary = {"total": len(retry_dict), "successful": 0, "skipped": 0, "failed": 0}
        
        for attempt in range(1, max_retries + 1):
            if not self.input_dict:
                break
                
            logger.info(f"\n=== Retry attempt {attempt}/{max_retries} ===")
            
            result = self.create_tiles()
            
            retry_summary["successful"] += result["successful"]
            retry_summary["skipped"] += result["skipped"]
            
            still_failed = []
            for input_item in self.input_dict:
                image_name = Path(input_item["image"]).stem
                if self.manifest.is_image_failed(image_name):
                    still_failed.append(input_item)
            
            self.input_dict = still_failed
            
            if not still_failed:
                logger.info(f"All images processed successfully after {attempt} attempt(s)")
                break
            
            logger.info(f"{len(still_failed)} images still failing after attempt {attempt}")
            
            if attempt < max_retries:
                wait_time = min(2 ** attempt, 30)
                logger.info(f"Waiting {wait_time} seconds before next retry...")
                time.sleep(wait_time)
        
        retry_summary["failed"] = len(self.input_dict)
        
        self.input_dict = original_input_dict
        
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
    
    @log_stage(stage_name="create_summary_visualization", log_memory=True)
    def create_summary_visualization(self, output_dir, prefix, samples_per_split=5):
        create_dataset_summary_visualization(output_dir, prefix, samples_per_split=samples_per_split)
    
    @log_stage(stage_name="process_single_pair", log_memory=True)
    def process_single_pair(self, image_path, label_path, tmp_dir):
        """Process a single image-label pair and return the result status."""
        image_name = Path(image_path).stem
        try:
            image_path = validate_image(image_path, self.bands_requested, self.band_indices)
            label_path, label_type = validate_mask(label_path)
            validation_result = validate_pair(image_path, label_path, label_type)
            if not validation_result["valid"]:
                return {"status": "skipped", "reason": validation_result["reason"]}
            special_case = validation_result.get("special_case", False)
            if special_case:
                return {"image_path": str(image_path), 
                        "label_path": str(label_path), 
                        "status": "successful", 
                        "reason": validation_result["reason"]}
            
            # Standard processing: CRS, overlap, intersection
            image_path, label_path = ensure_crs_match(image_path, label_path, label_type, tmp_dir)
            overlap_pct, overlap_msg = calculate_overlap(image_path, label_path, label_type)
            if overlap_pct == 0:
                return {"status": "skipped", "reason": f"No overlap: {overlap_msg}"}
            logger.info(f"Image: {image_name} with {overlap_msg}")
            clipped_image_path, clipped_label_path = clip_to_intersection(image_path, label_path, label_type, tmp_dir)
            log_memory_usage("After clip_to_intersection", force_gc=True)
            if clipped_image_path is None and clipped_label_path is None:
                return {"status": "skipped", "reason": "No intersection between image and label"}
            if label_type == 'vector':
                clipped_label_path = prepare_vector_labels(clipped_label_path,
                                                        clipped_image_path,
                                                        tmp_dir,
                                                        self.attr_field,
                                                        self.attr_values)

            return {"image_path": str(clipped_image_path), 
                    "label_path": str(clipped_label_path), 
                    "status": "successful", 
                    "reason": "Processed successfully"}
        except Exception as e:
            logger.error(f"Error processing image {image_name}: {e}")
            return {"status": "failed", "reason": str(e)}
    
    @log_stage(stage_name="process_analysis", log_memory=True)
    def process_analysis(self, image_path, label_path, metadata, image_name, sensor_type, 
                     image_analyses, global_class_distribution):
        """Process class distribution and spatial grid for an image-label pair and store results."""
        start_time = time.time()
        class_distribution = calculate_class_distribution(label_path, self.class_ids)
        class_distribution_time = time.time() - start_time
        logger.info(f"Class distribution calculated in {class_distribution_time:.2f} seconds")
        start_time = time.time()
        grid = create_spatial_grid(image_path, label_path, self.stride, self.grid_size, self.class_ids)
        grid_time = time.time() - start_time
        logger.info(f"Spatial grid created in {grid_time:.2f} seconds")
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
            
    def _get_or_create_writer(self, prefix, split, output_dir):
        """Get existing writer or create new one."""
        if prefix not in self.prefix_writers:
            self.prefix_writers[prefix] = {}
        
        if split not in self.prefix_writers[prefix]:
            shard_idx = self.prefix_shard_indices[prefix][split]
            shard_path = self._get_shard_path(output_dir, prefix, split, shard_idx)
            file_obj = open(shard_path, 'wb')
            writer = wds.TarWriter(file_obj)
            self.prefix_writers[prefix][split] = {'writer': writer,
                                                  'file_obj': file_obj,
                                                  'path': shard_path,
                                                  'start_size': 0}
            
        return self.prefix_writers[prefix][split]['writer']
    
    def _close_writer(self, prefix, split, flush_only=False):
        """Close or flush a specific writer"""
        if prefix not in self.prefix_writers or split not in self.prefix_writers[prefix]:
            return
        writer_info = self.prefix_writers[prefix][split]
        writer = writer_info['writer']
        file_obj = writer_info['file_obj']
        
        if flush_only:
            if hasattr(writer, 'tarfile') and hasattr(writer.tarfile, 'fileobj'):
                writer.tarfile.fileobj.flush()
                os.fsync(writer.tarfile.fileobj.fileno())
        else:
            writer.close()
            file_obj.close()
            del self.prefix_writers[prefix][split]
    
    def _close_all_writers(self, flush_only=False):
        """Close or flush all writers"""
        for prefix in list(self.prefix_writers.keys()):
            for split in list(self.prefix_writers[prefix].keys()):
                self._close_writer(prefix, split, flush_only)
        if not flush_only:
            self.prefix_writers.clear()
            
    @log_stage(stage_name="tiling", log_memory=True)
    def tiling(self,
                image_path: str,
                label_path: str,
                image_analysis: Dict[str, Any],
                validation_cells: List[str] = None,
                create_val_set: bool = False):
        try:       
            with rasterio.open(image_path) as src_image:
                with rasterio.open(label_path) as src_label:
                    image_width = src_image.width
                    image_height = src_image.height
                    image_bands = src_image.count
                    
                    label_width = src_label.width
                    label_height = src_label.height
                    label_bands = src_label.count
                    assert (image_width == label_width and 
                            image_height == label_height), "Image and label dimensions must match"
                    assert (self.patch_size[0] <= image_height and 
                            self.patch_size[1] <= image_width), "Patch size must be smaller than image dimensions"
            
                    metadata = image_analysis['metadata']
                    metadata["image_channels"] = image_bands
                    metadata["label_channels"] = label_bands
            
                    total_patches = image_analysis['grid']['total_patches']
                    grid_size = image_analysis['grid']['grid_size']
                    image_name = image_analysis['image_name']
                    
                    self.manifest.update_image_metadata(image_name, 
                                                        {"path": image_path,
                                                        "label_path": label_path,
                                                        "metadata": metadata,
                                                        "sensor_type": metadata.get("collection", "unknown"),
                                                        "class_distribution":image_analysis.get('class_distribution', {})
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
            
                    MAX_SHARD_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
                    discarded_count = 0
                    patch_count = 0
                    trn_patch_count = 0
                    val_patch_count = 0
                    tst_patch_count = 0
                    start_time = time.time()
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
                                
                                label_patch = src_label.read(window=window, boundless=False)
                                if not self._filter_patches(label_patch):
                                    discarded_count += 1
                                    continue
                                
                                image_patch = src_image.read(window=window, boundless=False)
                                
                                if image_patch.shape[1:] != self.patch_size or label_patch.shape[1:] != self.patch_size:
                                    image_patch = self.pad_patch(image_patch, self.patch_size)
                                    label_patch = self.pad_patch(label_patch, self.patch_size)
                                
                                if create_val_set:
                                    grid_x = int(x // (image_width / grid_size))
                                    grid_y = int(y // (image_height / grid_size))
                                    cell_id = f"{grid_x}_{grid_y}"
                                    if cell_id in validation_cells:
                                        split = "val"
                                        val_patch_count += 1
                                    else:
                                        split = "trn"
                                        trn_patch_count += 1
                                else:
                                    split = "tst"
                                    tst_patch_count += 1
                                
                                patch_key = f"{self.prefix}_{image_name}_{x}_{y}"
                                
                                all_metadata = {"patch_metadata": {"patch_id": patch_key,
                                                                    "pixel_coordinates": [x, y],
                                                                    "patch_size": self.patch_size,
                                                                    "stride": self.stride,
                                                                    "split": split,
                                                                    "image_dtype": src_image.dtypes[0],
                                                                    "label_dtype": src_label.dtypes[0],
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
                                current_shard_size = self._get_actual_shard_size(self.prefix, split)
                                estimated_patch_size = self._estimate_patch_size(image_patch, label_patch, all_metadata)
                                if current_shard_size + estimated_patch_size > MAX_SHARD_SIZE_BYTES:
                                    self._close_writer(self.prefix, split)
                                    self.manifest.close_shard(self.prefix, split,
                                                            self.prefix_shard_indices[self.prefix][split])
                                    self.prefix_shard_indices[self.prefix][split] += 1
                                    
                                    self.manifest.update_shard_record(self.prefix, split, 
                                                                        self.prefix_shard_indices[self.prefix][split], 
                                                                        0, 0, "OPEN", [image_name])
                                    logger.debug(f"Rotating to new {split} shard index: {self.prefix_shard_indices[self.prefix][split]}")
                                    self.prefix_shard_sizes[self.prefix][split] = 0
                                    
                                if split == "trn":
                                    writer = self._get_or_create_writer(self.prefix, split, output_train_dir)
                                elif split == "val":
                                    writer = self._get_or_create_writer(self.prefix, split, output_val_dir)
                                else: 
                                    writer = self._get_or_create_writer(self.prefix, split, output_tst_dir)
                                writer.write({"__key__": patch_key,
                                            "image_patch.npy": image_patch,
                                            "label_patch.npy": label_patch,
                                            "metadata.json": all_metadata})
                                actual_size = self._get_actual_shard_size(self.prefix, split)
                                self.prefix_shard_sizes[self.prefix][split] = actual_size
                                self.manifest.mark_patch_completed(image_name, x, y)
                                self.prefix_patch_counts[self.prefix][split] += 1
                                self.manifest.update_shard_info(self.prefix, 
                                                                split, 
                                                                self.prefix_shard_indices[self.prefix][split],
                                                                self.prefix_shard_sizes[self.prefix][split],
                                                                self.prefix_patch_counts[self.prefix][split])
                                self.manifest.update_image_patch_info(image_name, split, 
                                                                    self.prefix_shard_indices[self.prefix][split])
                                if patch_count % 100 == 0:
                                    self.manifest.save_manifest()
                                patch_count += 1
                                pbar.update(1)   
                    logger.info(f"""
                                Tiling Complete for {image_name}!
                                Training patches: {trn_patch_count}
                                Validation patches: {val_patch_count}
                                Test patches: {tst_patch_count}
                                ------------------------------
                                Extracted patches: {patch_count}
                                Discarded patches: {discarded_count}
                                Total patches: {total_patches}
                                """)
        except Exception as e:
            self.manifest.save_manifest()
            logger.error(f"Tiling failed: {e}")
        finally:
            end_time = time.time()
            image_name = image_analysis['image_name']
            image_tmp_dir = Path(self.output_dir) / self.prefix / "tmp" / image_name
            if image_tmp_dir.exists() and image_tmp_dir.is_dir():
                logger.info(f"Removing temporary directory: {image_tmp_dir}")
                shutil.rmtree(image_tmp_dir)
            logger.info(f"Tiling time: {end_time - start_time:.2f} seconds")
       
    def _filter_patches(self, label: np.ndarray) -> bool:
        """Filters patches based on the discard_empty flag and label_threshold."""
        if label.size == 0:
            logger.debug("Patch discarded: invalid shape or empty")
            return False
        nonzero_count = np.count_nonzero(label)
        if self.discard_empty and nonzero_count == 0:
            logger.debug("Patch discarded: all label values are 0")
            return False
        if self.label_threshold is not None:
            label_coverage = nonzero_count / label.size
            if label_coverage < self.label_threshold:
                logger.debug(f"Patch discarded: label coverage {label_coverage:.2f} < {self.label_threshold}")
                return False
        return True
    
    def _get_shard_path(self, base_path, prefix, split, idx):
        return os.path.join(base_path, f"{prefix}-{split}-{idx:06d}.tar")
    
    def _estimate_patch_size(self, image_patch, label_patch, metadata):
        size = image_patch.nbytes + label_patch.nbytes
        size += len(json.dumps(metadata).encode("utf-8"))
        return size
    
    def _get_actual_shard_size(self, prefix, split):
        """Get the actual current size of a shard file."""
        if prefix not in self.prefix_writers or split not in self.prefix_writers[prefix]:
            return 0
        writer_info = self.prefix_writers[prefix][split]
        writer = writer_info['writer']
        if hasattr(writer, 'tarfile') and hasattr(writer.tarfile, 'fileobj'):
            current_pos = writer.tarfile.fileobj.tell()
            return current_pos
        if os.path.exists(writer_info['path']):
            return os.path.getsize(writer_info['path'])
        
        return 0

    @staticmethod
    def pad_patch(patch: np.ndarray, patch_size: Tuple[int, int], mode='edge'):
        """Pads the patch to the patch size."""
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