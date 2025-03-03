import logging
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box
from typing import Tuple, Dict, List
from pathlib import Path

from utils.io import load_image, load_mask
from utils.geoutils import create_nodata_mask, apply_nodata_mask, rasterize_vector, get_intersection
from utils.geoutils import clip_to_intersection
from utils.checks import check_image_validity, check_label_validity, calculate_overlap
from utils.checks import is_image_georeferenced, is_label_georeferenced, check_alignment
from config.logging_config import logger

logger = logging.getLogger(__name__)


class Tiler:
    
    def __init__(self, 
                 input_pairs: List[Tuple[str, str]], 
                 patch_size: Tuple[int, int], # (height, width)
                 attr_field: str,
                 attr_values: List[str],
                 stride: int = None,
                 discard_empty: bool = True,
                 label_threshold: float = 0.01, # minimum of non-zero pixels in a patch to be considered valid (0-1)
                 single_class_mode: bool = False,
                 multiclass_mode: Dict[str, bool] = None,
                 write_label_raster: bool = False,
                 label_raster_path: str = None,
                 output_dir: str = None):
        
        self.input_pairs = input_pairs
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.discard_empty = discard_empty
        self.label_threshold = label_threshold
        self.single_class_mode = single_class_mode
        self.multiclass_mode = multiclass_mode
        self.output_dir = output_dir
        self.attr_field = attr_field
        self.attr_values = attr_values
        self.write_label_raster = write_label_raster
        self.label_raster_path = label_raster_path
        
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
    
    
    def _filter_patches(self, image: np.ndarray, label: np.ndarray):
        """Filters patches based on the discard_empty flag and label_threshold"""
        if self.discard_empty and np.all(label) == 0:
            return False
        
        if self.label_threshold:
            label_percentage = np.sum(label > 0) / label.size
            if label_percentage < self.label_threshold:
                return False
        return True
    
    
    def _prepare_vector_labels(self, label: gpd.GeoDataFrame, image: rasterio.DatasetReader):
        """Prepares vector labels for tiling"""
        nodata_mask = create_nodata_mask(image)
        label = apply_nodata_mask(label, nodata_mask)
        label = rasterize_vector(label, image,
                                  attr_field=self.attr_field,
                                  attr_values=self.attr_values,
                                  write_raster=self.write_label_raster, 
                                  output_path=self.label_raster_path)
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
        
        for y in range(0, image_height, self.stride):
            for x in range(0, image_width, self.stride):
                window_width = min(self.patch_size[1], image_width - x)
                window_height = min(self.patch_size[0], image_height - y)
                
                window = rasterio.windows.Window(col_off=x, row_off=y, width=window_width, height=window_height)
                image_patch = image.read(window=window, boundless=False)
                label_patch = label.read(window=window, boundless=False)
                
                if window_width < self.patch_size[1] or window_height < self.patch_size[0]:
                    image_patch = self.pad_patch(image_patch, self.patch_size)
                    label_patch = self.pad_patch(label_patch, self.patch_size)
                
                if self._filter_patches(image_patch, label_patch):
                    image_patches.append(image_patch)
                    label_patches.append(label_patch)
        
        return image_patches, label_patches
    
    def _build_output_folder(self, image_name: str, id: int):
        """Builds the output path for the tiles"""
        image_folder = f"{image_name}_{id}"
        output_folder = Path(self.output_dir) / image_folder
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)
        return output_folder
    
    def _build_patch_path(self, output_folder: Path, 
                          image_name: str, x: int, y: int):
        """Builds the output path for the tiles"""
        patch_name = f"{image_name}_{x}_{y}.tif"
        return output_folder / patch_name
    
    def create_tiles(self):
        """Creates tiles from input pairs"""
        
        for input_pair in self.input_pairs:
            
            image_path, label_path = input_pair
            image_name = Path(image_path).stem
            output_folder = self._build_output_folder(image_name, id)
            
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
                    logger.info("Tiling image and label pair")
                    self.tiling(image, label)
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
            
            image_patches, label_patches = self.tiling(image, label)
    


if __name__ == '__main__':
    data = [("https://int.datacube.services.geo.ca/stac/api/collections/worldview-2-ortho-pansharp/items/ON_Gore-Bay_WV02_20110828", "/home/valhassa/Projects/geotiff-tiler/data/ON45.gpkg"), 
            ("/home/valhassa/Projects/geotiff-tiler/data/AB26_NRGB_8bit_clahe25.tif", "/home/valhassa/Projects/geotiff-tiler/data/AB26.gpkg"),
            ("/home/valhassa/Projects/geotiff-tiler/data/GF2_PMS1__L1A0000564539-MSS1.tif", "/home/valhassa/Projects/geotiff-tiler/data/GF2_PMS1__L1A0000564539-MSS1_24label.tif")]
    
    tiler = Tiler(input_pairs=[data[1]], 
                  tile_size=(1024, 1024),
                  attr_field=["class", "Quatreclasses"],
                  attr_values=[1, 2, 3, 4],
                  stride=(512, 512), 
                  discard_empty=True, 
                  label_threshold=0.5, 
                  single_class_mode=False, 
                  multiclass_mode={'class1': True, 'class2': False}, 
                  output_dir='/home/valhassa/Projects/geotiff-tiler/data/output')
    
    tiler.create_tiles()
    
    
    
