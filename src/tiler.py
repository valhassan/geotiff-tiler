import pystac
import logging
import rasterio
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
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
                 tile_size: Tuple[int, int],
                 attr_field: str,
                 attr_values: List[str],
                 stride: Tuple[int, int],
                 discard_empty: bool,
                 label_threshold: float,
                 single_class_mode: bool,
                 multiclass_mode: Dict[str, bool],
                 write_label_raster: bool,
                 label_raster_path: str,
                 output_dir: str):
        
        self.input_pairs = input_pairs
        self.tile_size = tile_size
        self.stride = stride
        self.discard_empty = discard_empty
        self.label_threshold = label_threshold
        self.single_class_mode = single_class_mode
        self.multiclass_mode = multiclass_mode
        self.output_dir = output_dir
        self.attr_field = attr_field
        self.attr_values = attr_values
        self.write_label_raster = write_label_raster
        self.label_raster_path = label_raster_path
        
        
    
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
        
        
    
    # def _prepare_raster_labels(self, label: rasterio.DatasetReader, image: rasterio.DatasetReader):
    #     """Prepares raster labels for tiling"""
    #     nodata_mask = create_nodata_mask(image)
    #     label = apply_nodata_mask_raster(label, nodata_mask, self.label_raster_path)
    
    def tiling(self, image: rasterio.DatasetReader, label: rasterio.DatasetReader):
        """Tiles the image and label"""
        # intersection = image.read(1) & label.read(1)
        pass
    
    def create_tiles(self):
        """Creates tiles from input pairs"""
        
        for input_pair in self.input_pairs:
            image_path, label_path = input_pair
            image = load_image(image_path)
            label = load_mask(label_path)
            if isinstance(label, gpd.GeoDataFrame) and (not is_image_georeferenced(image) or 
                    not is_label_georeferenced(label)):
                logger.info("Skipping pair due to invalid label or image")
                continue
            
            if isinstance(label, rasterio.DatasetReader) and (not is_image_georeferenced(image) or 
                                                              not is_label_georeferenced(label)):
                if check_alignment(image, label):
                    logger.info("Tiling image and label pair")
                    self.tiling(image, label)
                else:
                    logger.info("Skipping pair due to invalid label or image")
                    continue
            
            else:
                image_valid, image_msg = check_image_validity(image)
                if not image_valid:
                    print(f"Skipping pair due to invalid image: {image_msg}")
                    continue
                
                label_valid, label_msg = check_label_validity(label)
                if not label_valid:
                    print(f"Skipping pair due to invalid label: {label_msg}")
                    continue
                
                overlap_pct, overlap_msg = calculate_overlap(image, label)
                if overlap_pct == 0:
                    print(f"Skipping pair due to no overlap: {overlap_msg}")
                    continue
                else:
                    print(f"Processing pair with {overlap_msg}")
                
                intersection = get_intersection(image, label)
                if intersection is None:
                    logger.info("Skipping pair: No intersection between image and label")
                    continue
                clipped_image, clipped_label = clip_to_intersection(image, label, intersection)
                
                print(f"image.crs: {image.crs}")
                print(f"image.bounds: {box(*image.bounds)}")
                print(f"image.count: {image.count}")
                print(f"image.height: {image.height}")
                print(f"image.width: {image.width}")
                
                print(f"clipped_image.crs: {clipped_image.crs}")
                print(f"clipped_image.bounds: {box(*clipped_image.bounds)}")
                print(f"clipped_image.count: {clipped_image.count}")
                print(f"clipped_image.height: {clipped_image.height}")
                print(f"clipped_image.width: {clipped_image.width}")
                
                
                if isinstance(clipped_label, gpd.GeoDataFrame):
                    
                    label_bounds = clipped_label.total_bounds
                    print(f"label_bounds: {box(*label_bounds)}")
                    print(f"label.crs: {clipped_label.crs}")
                    label = self._prepare_vector_labels(clipped_label, clipped_image)
                    
                    print(f"____label.bounds: {box(*label.bounds)}")
                    
                
            # Continue with tiling...
    


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
                  output_dir='/home/valhassa/Projects/geotiff-tiler/data/output',
                  write_label_raster=False,
                  label_raster_path='label.tif')
    
    tiler.create_tiles()
    
    
    
