import pystac
import rasterio
import fiona
import geopandas as gpd
from pathlib import Path
from typing import Sequence
from .checks import check_stac, check_label_type
from .stacitem import SingleBandItemEO
from .geoutils import stack_bands, select_bands

def load_image(image_path: str, 
               bands_requested: Sequence = ["red", "green", "blue"], 
               band_indices: Sequence | None = None):
    
    """Loads an image from a path or stac item"""
    
    image_asset = None
    
    if check_stac(image_path):
        item = SingleBandItemEO(item=pystac.Item.from_file(image_path),
                                bands_requested=bands_requested)
        stac_bands = [value['meta'].href for value in item.bands_requested.values()]
        image_asset = stack_bands(stac_bands)
    
    elif Path(image_path).exists():
        if band_indices:
            image_asset = select_bands(image_path, band_indices)
        else:
            image_asset = image_path
    
    if image_asset:
        raster = rasterio.open(image_asset)
        return raster
    else:
        raise FileNotFoundError(f"File not found: {image_path}")

def load_vector(vector: gpd.GeoDataFrame, skip_layer: str | None = None):
    
    pass


def load_mask(mask_path: str, skip_layer: str | None = "extent"):
    """Loads a mask from a path. Accepts a local tiff or json file"""
    if Path(mask_path).exists():
        label_type = check_label_type(mask_path)
        if label_type == 'raster':
            return rasterio.open(mask_path)
        elif label_type == 'vector':
            layers = fiona.listlayers(mask_path)
            extent_layer = next((layer for layer in layers if "extent" in layer.lower()), None)
            main_layer = next(
                (layer for layer in layers if skip_layer not in layer.lower()), None)
            if main_layer is None:
                raise ValueError(f"No suitable layer found in {mask_path}")
            result = gpd.read_file(mask_path, layer=main_layer)
            if extent_layer:
                extent_gdf = gpd.read_file(mask_path, layer=extent_layer)
                if not extent_gdf.empty:
                    result.attrs['extent_geometry'] = extent_gdf.geometry.iloc[0]
            return result
    else:
        raise FileNotFoundError(f"File not found: {mask_path}")




if __name__ == '__main__':
    stac_image = 'https://int.datacube.services.geo.ca/stac/api/collections/worldview-2-ortho-pansharp/items/ON_Gore-Bay_WV02_20110828'
    image_path = 'data/worldview-2-ortho-pansharp/ON_Gore-Bay_WV02_20110828_1030010019004C00.tif'
    bands_requested = ["red", "green", "blue"]
    raster = load_image(stac_image, bands_requested)
    print(raster.meta)
    raster.close()
    