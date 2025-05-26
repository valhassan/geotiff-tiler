# GeoTIFF Tiler

A Python package for creating training patches from geospatial imagery and label pairs for machine learning applications.

## Overview

GeoTIFF Tiler is designed to streamline the creation of training data patches from geo-referenced and non-geo-referenced image and label pairs. It helps prepare data for machine learning models requiring consistent input dimensions, particularly for geospatial applications. The library supports modern cloud-native geospatial workflows and provides robust data management capabilities.

## Features

- **Multi-format Input Support**:
  - **Images**: GeoTIFFs (geo-referenced and non-geo-referenced), STAC imagery, cloud-optimized GeoTIFFs
  - **Labels**: GeoTIFFs (geo-referenced and non-geo-referenced), vector data (.geojson, .gpkg, .shp)
- **WebDataset Output Format**: Efficient sharded format for distributed training
- **Intelligent Data Splitting**:
  - Spatial validation splitting with configurable grid-based selection
  - Class-balanced validation sets with customizable weights
  - Automatic handling of train/validation/test splits
- **Advanced Patch Management**:
  - Intelligent patch filtering based on label content and thresholds
  - Padding for edge patches to maintain consistent dimensions
  - Automatic handling of CRS and alignment issues
  - Memory-efficient processing with resource management
- **STAC Integration**: Native support for SpatioTemporal Asset Catalog items
- **Robust Processing**:
  - Resumable operations with comprehensive manifest system
  - Automatic retry mechanisms for failed processing
  - Progress tracking and detailed logging
  - Memory monitoring and optimization
- **Multi-sensor Support**:
  - Band selection and mapping for different sensor types
  - Automatic normalization statistics calculation
- **Quality Assessment Tools**:
  - WebDataset-compatible visualization functions
  - Automatic visualization generation during processing
  - Dataset summary visualizations across splits
  - Per-image patch visualization with metadata
  - Class distribution analysis and validation reports

## Installation

```bash
pip install geotiff-tiler
```

## Quick Start

### Basic Usage

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
    patch_size=(256, 256),                            # Height, Width
    bands_requested=["red", "green", "blue", "nir"],  # Band selection
    stride=128,                                       # Overlap between patches
    discard_empty=True,                               # Skip patches with no labels
    label_threshold=0.05,                             # Minimum non-zero label coverage
    output_dir='./output/patches',
    prefix='dataset_v1'                               # Dataset identifier
)

# Create the patches
tiler.create_tiles()
```

### STAC Integration

The library supports STAC (SpatioTemporal Asset Catalog) items, making it compatible with cloud-native geospatial workflows:

```python
# Using STAC items directly
data = [{
    "image": "https://stac-api.example.com/collections/sentinel-2/items/item-id",
    "label": "./path/to/label.geojson",
    "metadata": {"collection": "sentinel-2", "gsd": 10.0}
}]

tiler = Tiler(
    input_dict=data,
    patch_size=(512, 512),
    bands_requested=["red", "green", "blue", "nir"],
    attr_field=["class"],       # Vector label field
    attr_values=[1, 2, 3, 4],   # Class values to extract
    output_dir='./stac_patches'
)
```

### Advanced Configuration

```python
tiler = Tiler(
    input_dict=data,
    patch_size=(1024, 1024),
    bands_requested=["red", "green", "blue", "nir"],
    stride=512,
    
    # Validation splitting parameters
    grid_size=8,                    # Spatial grid for validation selection
    val_ratio=0.2,                  # 20% for validation
    class_balance_weight=0.6,       # Weight for class balance in validation
    spatial_weight=0.4,             # Weight for spatial coverage in validation
    
    # Label processing
    attr_field=["class", "category"],  # Fields in vector data
    attr_values=[1, 2, 3, 4],          # Values to extract
    class_ids={                        # Custom class mapping
        'background': 0,
        'water': 1,
        'vegetation': 2,
        'urban': 3,
        'bare_soil': 4
    },
    
    # Quality control
    discard_empty=True,
    label_threshold=0.1,            # 10% minimum label coverage
    
    # Output configuration
    prefix='landcover_v1',
    output_dir='./datasets/landcover'
)

result = tiler.create_tiles()
print(f"Processing complete: {result}")
```

## Parameters

### Core Parameters
- **input_dict**: List of dictionaries with "image", "label", and "metadata" keys
- **patch_size**: Tuple of (height, width) for the output patches
- **bands_requested**: List of band names to extract (e.g., ["red", "green", "blue", "nir"])
- **stride**: Spacing between patches (determines overlap); if None, uses max(patch_size)
- **output_dir**: Directory to save the output patches
- **prefix**: Dataset identifier for output files
- **create_viz**: Whether to automatically create visualizations for completed images (default: False)

### Label Processing
- **attr_field**: Field name(s) in vector data to use for labeling (list of strings)
- **attr_values**: Values to extract from the attribute field (list of strings or numbers)
- **class_ids**: Dictionary mapping class names to numeric IDs
- **discard_empty**: Whether to skip patches with no labels
- **label_threshold**: Minimum fraction of non-zero pixels required in a label patch

### Validation Splitting
- **grid_size**: Size of spatial grid for validation selection (default: 4)
- **val_ratio**: Fraction of data to use for validation (default: 0.2)
- **class_balance_weight**: Weight for class balance in validation selection (default: 0.5)
- **spatial_weight**: Weight for spatial coverage in validation selection (default: 0.5)

## Output Format

The tiler creates datasets in WebDataset format with the following structure:

```
output_dir/
├── prefix_manifest.json           # Processing manifest and statistics
├── normalization_stats.json       # Band statistics for normalization
├── prefix/
│   ├── trn/                       # Training shards
│   │   ├── prefix_000000.tar
│   │   ├── prefix_000001.tar
│   │   └── ...
│   ├── val/                       # Validation shards
│   │   ├── prefix_000000.tar
│   │   └── ...
│   ├── tst/                       # Test shards (if applicable)
│   │   └── ...
│   └── viz/                       # Visualization outputs (if create_viz=True)
│       ├── trn/
│       │   ├── image1_trn.png
│       │   └── ...
│       └── val/
│           ├── image1_val.png
│           └── ...
```

Each shard contains:
- **Image patches**: `.npy` format
- **Label patches**: `.npy` format
- **Metadata**: `.json` format with spatial and processing information

## Advanced Features

### Resumable Operations

The tiler automatically saves progress and can resume interrupted processing:

```python
# If processing is interrupted, simply run again with the same configuration
tiler = Tiler(input_dict=data, output_dir='./output', prefix='dataset_v1')
tiler.create_tiles()  # Will automatically resume from where it left off
```

### Retry Failed Images

```python
# Retry processing of failed images
retry_results = tiler.retry_failed_images(max_retries=3)
print(f"Retry results: {retry_results}")
```

### Export Statistics

```python
# Export normalization statistics for model training
tiler.export_normalization_stats('./model_stats.json')
```

### Visualization

```python
from geotiff_tiler.utils.visualization import visualize_webdataset_patches, create_dataset_summary_visualization

# Visualize patches from a specific image
visualize_webdataset_patches(
    dataset_dir='./output',
    prefix='dataset_v1',
    split='trn',
    image_name='specific_image_name',  # Optional: target specific image
    n_samples=5,
    save_path='./visualization.png'
)

# Create comprehensive dataset summary across all splits
create_dataset_summary_visualization(
    output_dir='./output',
    prefix='dataset_v1',
    samples_per_split=6,
    images_per_split=3
)

# Enable automatic visualization during processing
tiler = Tiler(
    input_dict=data,
    patch_size=(256, 256),
    create_viz=True,  # Automatically create visualizations for completed images
    output_dir='./output'
)
```

## Requirements

See `requirements.txt` for complete dependency list.

## License

MIT License

## Author

Victor Alhassan (victor.alhassan@nrcan-rncan.gc.ca)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
