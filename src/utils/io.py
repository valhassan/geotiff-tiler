import logging
import zarr
import pystac
import rasterio
import fiona
import numpy as np
import geopandas as gpd
from pprint import pprint
from pathlib import Path
from typing import Sequence
from .checks import check_stac, check_label_type
from .stacitem import SingleBandItemEO
from .geoutils import stack_bands, select_bands

logger = logging.getLogger(__name__)

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

def save_patches_to_zarr(image_patches: list[np.ndarray],
                         label_patches: list[np.ndarray],
                         patch_locations: list[tuple[int, int]],
                         metadata: dict,
                         output_dir: str,
                         image_name: str) -> Path | None:
        """
        Save image and label patches to Zarr 3.0 format without sharding, optimized for performance.

        Parameters:
        -----------
        image_patches : List[np.ndarray]
            List of image patches (e.g., shape: [channels, height, width]).
        label_patches : List[np.ndarray]
            List of label patches (e.g., shape: [classes, height, width]).
        patch_locations : List[Tuple[int, int]]
            List of (x, y) locations for each patch.
        image_name : str
            Name of the original image (used for naming the Zarr store).

        Returns:
        --------
        zarr_path : Path | None
            Path to the created Zarr store, or None if no patches are provided.
        """
        # Validate input
        if not image_patches or not label_patches or len(image_patches) != len(label_patches):
            logger.warning(f"No valid patches to save for {image_name}")
            return None

        try:
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine shapes and dtypes
            n_patches = len(image_patches)
            image_shape = image_patches[0].shape  # e.g., (channels, height, width)
            label_shape = label_patches[0].shape  # e.g., (classes, height, width)

            # Validate patch consistency
            if not all(p.shape == image_shape for p in image_patches) or \
               not all(p.shape == label_shape for p in label_patches):
                raise ValueError("Inconsistent patch shapes detected")

            # Stack patches into arrays
            stacked_images = np.stack(image_patches, axis=0)  # (n_patches, channels, h, w)
            stacked_labels = np.stack(label_patches, axis=0)  # (n_patches, classes, h, w)
            patch_locations_array = np.array(patch_locations)  # (n_patches, 2)

            # Define Zarr store path
            zarr_path = output_dir / f"{image_name}.zarr"

            # Initialize store with Zarr V3 protocol
            store = zarr.storage.LocalStore(str(zarr_path))
            root = zarr.group(store=store)

            # Use Blosc compression for efficiency
            compressor = zarr.codecs.BloscCodec(cname='lz4', clevel=5, shuffle='shuffle')

            # Store images with one patch per chunk
            images_array = root.create_array(name='images',
                                             shape=stacked_images.shape,
                                             chunks=(1, *image_shape),  # Each patch is a chunk
                                             compressor=compressor,
                                             dtype=stacked_images.dtype)
            images_array[:] = stacked_images
            
            # Store labels with one patch per chunk
            labels_array = root.create_array(name='labels',
                                             shape=stacked_labels.shape,
                                             chunks=(1, *label_shape),  # Each patch is a chunk
                                             compressor=compressor,
                                             dtype=stacked_labels.dtype)
            labels_array[:] = stacked_labels

            # Store patch locations as a single chunk (small data)
            locations_array = root.create_array(name='locations',
                                                shape=patch_locations_array.shape,
                                                chunks=(n_patches, 2),  # Single chunk for simplicity
                                                compressor=compressor,
                                                dtype=patch_locations_array.dtype)
            locations_array[:] = patch_locations_array
            
            metadata["image_channels"] = image_shape[0]
            metadata["label_channels"] = label_shape[0]
            root.attrs.update(metadata)

            logger.info(f"Saved {n_patches} patches to {zarr_path} using Zarr")
            return zarr_path

        except Exception as e:
            logger.error(f"Failed to save patches for {image_name}: {str(e)}")
            raise

def read_patches_from_zarr(zarr_path, indices=None):
        """
        Read patches from Zarr store
        
        Parameters:
        -----------
        zarr_path : str or Path
            Path to the Zarr store
        indices : list, slice, or None
            Indices of patches to read. If None, read all patches
            
        Returns:
        --------
        image_patches : np.ndarray
            Image patches with shape (n_patches, channels, height, width)
        label_patches : np.ndarray
            Label patches with shape (n_patches, classes, height, width)
        patch_locations : np.ndarray
            (x, y) coordinates of each patch
        """
        zarr_path = str(zarr_path)
        root = zarr.open(zarr_path, mode='r')
        # print(f"{root['images'].info}")
        print("Root Group Metadata:")
        pprint(dict(root.attrs))
        print("\nRoot Group Info:")
        print(root.info)
        
        if indices is None:
            image_patches = root['images'][:]
            label_patches = root['labels'][:]
            patch_locations = root['locations'][:]
        else:
            image_patches = root['images'][indices]
            label_patches = root['labels'][indices]
            patch_locations = root['locations'][indices]
        
        return image_patches, label_patches, patch_locations


if __name__ == '__main__':
    stac_image = 'https://int.datacube.services.geo.ca/stac/api/collections/worldview-2-ortho-pansharp/items/ON_Gore-Bay_WV02_20110828'
    image_path = 'data/worldview-2-ortho-pansharp/ON_Gore-Bay_WV02_20110828_1030010019004C00.tif'
    bands_requested = ["red", "green", "blue"]
    raster = load_image(stac_image, bands_requested)
    print(raster.meta)
    raster.close()
    