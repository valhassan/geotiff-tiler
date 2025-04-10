# GeoTIFF Tiler

A Python package for creating training patches from geospatial imagery and label pairs for machine learning applications.

## Overview

GeoTIFF Tiler is designed to streamline the creation of training data patches from geo-referenced and non-geo-referenced image and label pairs. It helps prepare data for machine learning models requiring consistent input dimensions, particularly for geospatial applications.

## Features

- Create patches of specified size from image-label pairs
- Support for various input formats:
  - **Images**: GeoTIFFs (geo-referenced and non-geo-referenced), STAC imagery
  - **Labels**: GeoTIFFs (geo-referenced and non-geo-referenced), vector data (.geojson, .gpkg, .shp)
- Intelligent patch filtering based on label content
- Padding for edge patches to maintain consistent dimensions
- Automatic handling of CRS and alignment issues
- Output in Zarr format for efficient storage and access
- Visualization tools for quality assessment

## Installation

```bash
pip install geotiff-tiler
```

## Quick Start

```python
from geotiff_tiler.tiler import Tiler

# Define your image-label pairs with metadata
data = [{
    "image": "./path/to/image.tif",
    "label": "./path/to/label.tif",
    "metadata": {"collection": "satellite-name", "gsd": 0.5}
}]

# Initialize the tiler with your configuration
tiler = Tiler(
    input_dict=data,
    patch_size=(256, 256),  # Height, Width
    attr_field="class",     # Field in vector data to use for labels
    attr_values=[1, 2, 3],  # Values to extract from the field
    stride=128,             # Overlap between patches
    discard_empty=True,     # Skip patches with no labels
    label_threshold=0.05,   # Minimum non-zero label coverage
    output_dir='./output/patches'
)

# Create the patches
tiler.create_tiles()
```

### Using STAC Items

The library supports STAC (SpatioTemporal Asset Catalog) items, making it compatible with cloud-native geospatial workflows.

## Parameters

- **input_dict**: List of dictionaries with "image", "label", and "metadata" keys
- **patch_size**: Tuple of (height, width) for the output patches
- **attr_field**: Field name(s) in vector data to use for labeling
- **attr_values**: Values to extract from the attribute field
- **stride**: Spacing between patches (determines overlap)
- **discard_empty**: Whether to skip patches with no labels
- **label_threshold**: Minimum fraction of non-zero pixels required in a label patch
- **output_dir**: Directory to save the output patches

## Output Format

Patches are saved in Zarr format with the following structure:
- `images`: Array of image patches [N, C, H, W]
- `labels`: Array of label patches [N, H, W]
- `positions`: Array of patch locations [N, 2]
- `metadata`: Dictionary with additional information

A csv file is created of the zarr paths.

## License

MIT License

## Author

Victor Alhassan (victor.alhassan@nrcan-rncan.gc.ca)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
