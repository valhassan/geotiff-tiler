import logging
import rasterio
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from rasterio.windows import Window
from typing import Tuple, Dict, List, Any
from pathlib import Path

from geotiff_tiler.utils.io import load_image, load_mask, save_patches_to_zarr
from geotiff_tiler.utils.geoutils import create_nodata_mask, apply_nodata_mask, rasterize_vector, get_intersection
from geotiff_tiler.utils.geoutils import ensure_crs_match, clip_to_intersection
from geotiff_tiler.utils.checks import check_image_validity, check_label_validity, calculate_overlap
from geotiff_tiler.utils.checks import is_image_georeferenced, is_label_georeferenced, check_alignment
from geotiff_tiler.utils.visualization import visualize_zarr_patches
from geotiff_tiler.config.logging_config import logger

logger = logging.getLogger(__name__)


class Tiler:
    
    def __init__(self, 
                 input_dict: List[Dict[str, Any]], 
                 patch_size: Tuple[int, int], # (height, width)
                 attr_field: str,
                 attr_values: List[str],
                 stride: int = None,
                 discard_empty: bool = True,
                 label_threshold: float = 0.01, # minimum of non-zero pixels in a patch to be considered valid (0-1)
                 output_dir: str = None):
        
        self.input_dict = input_dict
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.discard_empty = discard_empty
        self.label_threshold = label_threshold
        self.output_dir = output_dir
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
    
    def tiling(self, image: rasterio.DatasetReader, label: rasterio.DatasetReader):
        """Tiles the image and label"""
        
        image_width = image.width
        image_height = image.height
        
        label_width = label.width
        label_height = label.height
        
        assert (image_width == label_width and 
                image_height == label_height), "Image and label dimensions must match"
        assert (self.patch_size[0] <= image_height and 
                self.patch_size[1] <= image_width), "Patch size must be smaller than image dimensions"
        
        image_patches = []
        label_patches = []
        positions = []
        
        try:
            logger.info(f"Tiling {image_height} x {image_width} "
                        f"image with patch size {self.patch_size} and stride {self.stride}")
            
            for y in range(0, image_height, self.stride):
                for x in range(0, image_width, self.stride):
                    window_width = min(self.patch_size[1], image_width - x)
                    window_height = min(self.patch_size[0], image_height - y)
                    
                    window = Window(col_off=x, row_off=y, width=window_width, height=window_height)
                    
                    label_patch = label.read(window=window, boundless=False)
                    if not self._filter_patches(label_patch):
                        continue
                    image_patch = image.read(window=window, boundless=False)
                    
                    if image_patch.shape[1:] != self.patch_size or label_patch.shape[1:] != self.patch_size:
                        image_patch = self.pad_patch(image_patch, self.patch_size)
                        label_patch = self.pad_patch(label_patch, self.patch_size)
                    image_patches.append(image_patch)
                    label_patches.append(label_patch)
                    positions.append((x, y))
            logger.info(f"Extracted {len(image_patches)} patches")
            return image_patches, label_patches, positions
        except Exception as e:
            logger.error(f"Tiling failed: {e}")
            raise
    
    def _build_output_folder(self, image_name: str, id: int):
        """Builds the output path for the tiles"""
        id_str = str(id).zfill(3)
        image_folder = f"{id_str}_{image_name}"
        output_folder = Path(self.output_dir) / image_folder
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder
    
    def _build_patch_path(self, output_folder: Path, 
                          image_name: str, x: int, y: int):
        """Builds the output path for the tiles"""
        patch_name = f"{image_name}_{x}_{y}.tif"
        return output_folder / patch_name
    
    def _process_and_save_tiles(self, image, label, metadata, output_folder, image_name):
        """Processes tiles and saves them to zarr format"""
        image_patches, label_patches, positions = self.tiling(image, label)
        zarr_path = save_patches_to_zarr(image_patches=image_patches,
                                         label_patches=label_patches,
                                         patch_locations=positions,
                                         metadata=metadata,
                                         output_dir=output_folder,
                                         image_name=image_name)
        visualize_zarr_patches(zarr_path,
                               save_path=output_folder / f"{image_name}_patches.png")
        return zarr_path
    
    
    def create_tiles(self):
        """Creates tiles from input pairs"""
        
        try:
        
            for id, input_dict in tqdm(enumerate(self.input_dict), 
                                    total=len(self.input_dict), desc="Processing input pairs"):
                image_path = input_dict["image"]
                label_path = input_dict["label"]
                metadata = input_dict["metadata"]
                image_name = Path(image_path).stem
                output_folder = self._build_output_folder(image_name, id)
                
                metadata["image_name"] = image_name
                metadata["patch_size"] = self.patch_size
                metadata["stride"] = self.stride
                
                try:
                    image = load_image(image_path)
                    label = load_mask(label_path)
                except Exception as e:
                    logger.error(f"Error loading image or label: {e}")
                    continue
                
                if isinstance(label, gpd.GeoDataFrame) and (not is_image_georeferenced(image) or 
                        not is_label_georeferenced(label)):
                    logger.info("Skipping pair due to invalid label or image")
                    continue
                
                if isinstance(label, rasterio.DatasetReader) and (not is_image_georeferenced(image) or 
                                                                not is_label_georeferenced(label)):
                    if check_alignment(image, label):
                        self._process_and_save_tiles(image, label, metadata, output_folder, image_name)
                        continue
                    else:
                        logger.info("Skipping pair due to invalid label or image")
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
                
                if isinstance(label, gpd.GeoDataFrame):                    
                    label = self._prepare_vector_labels(label, image)
                
                self._process_and_save_tiles(image, label, metadata, output_folder, image_name)
                
                if 'image' in locals() and isinstance(image, rasterio.DatasetReader):
                    image.close()
                if 'label' in locals() and isinstance(label, rasterio.DatasetReader):
                    label.close()
            
            logger.info("Tiling complete")
        except Exception as e:
            logger.error(f"Tiling process failed: {e}")
            raise



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
    