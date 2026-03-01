# ITUR Urbanization Prediction from Sentinel-2 Satellite Images

This repository provides a machine learning pipeline to predict urbanization levels (ITUR values) from Sentinel-2A satellite images. The pipeline leverages deep learning with pretrained weights, advanced spectral indices, and robust training strategies such as early stopping.

## Key Features

- **Urbanization Prediction**: Predicts ITUR (urbanization) values from Sentinel-2A satellite imagery.
- **Flexible Channel Selection**: Supports both the 13 Sentinel-2 bands and a wide range of spectral indices (e.g., NDVI, NDBI, SAVI, MSAVI, NDWI, NMDI, NBR).
- **Pretrained Weights**: Utilizes TorchGeo/torchvision pretrained ResNet weights for improved performance and faster convergence.
- **Early Stopping**: Prevents overfitting by monitoring validation metrics and restoring the best model.
- **Fast CSV Workflow**: Skip slow image/label matching by using pre-matched CSVs for train, validation, and test splits.
- **Customizable**: Easily adjust which bands/indices to use, training parameters, and data splits.

## Workflow Overview

1. **Prepare Data**
   - Organize Sentinel-2 `.tif` images in a directory.
   - Prepare a CSV file with ITUR values and image codes.
   - (Recommended) Run the legacy matching workflow once to generate pre-matched CSVs for train/val/test splits.
2. **Configure Model**
   - Specify which bands and spectral indices to use via the `custom_channels` list.
   - Choose training parameters (epochs, batch size, learning rate, etc.).
3. **Run Training**
   - Use the fast workflow with pre-matched CSVs to skip matching and accelerate training.
   - The model will use pretrained weights and early stopping by default.
4. **Evaluate & Visualize**
   - Training/validation/test metrics and plots are saved in the `outputs/` directory.

## Quickstart

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Place your Sentinel-2 `.tif` images in a directory.
- Prepare a CSV with at least `image_path` and `itur_value` columns (optionally `code`).
- (Optional but recommended) Run the legacy workflow to generate pre-matched CSVs for train, val, and test splits:
  - `outputs/train_matched_data.csv`
  - `outputs/val_matched_data.csv`
  - `outputs/test_matched_data.csv`

### 3. Configure Channels and Indices

Edit the `custom_channels` list in `ML_Updated.py` to specify which bands and indices to use, e.g.:

```python
custom_channels = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12',
    'NDVI', 'NDBI', 'SAVI', 'MSAVI', 'NDWI', 'NMDI', 'NBR'
]
```

### 4. Run Training

```bash
python ML_Updated.py
```

- The script will use the pre-matched CSVs and your specified channels/indices.
- Model weights, metrics, and plots will be saved in the `outputs/` directory.

## File Structure

- `ML_Updated.py` — Main pipeline script (data loading, model, training, evaluation)
- `requirements.txt` — All required Python packages
- `outputs/` — Saved models, metrics, and plots
- `train_matched_data.csv`, `val_matched_data.csv`, `test_matched_data.csv` — Pre-matched CSVs for fast workflow

## Advanced Usage

- **Legacy Matching**: If you do not have pre-matched CSVs, the script can match images to labels using the original directory and CSV, but this is slower for large datasets.
- **Custom Indices**: You can add or remove spectral indices in `custom_channels` as needed for your research.
- **Early Stopping**: The model will automatically stop training if validation metrics do not improve.
- **Pretrained Weights**: The model uses TorchGeo/torchvision pretrained weights for Sentinel-2 or ImageNet by default.

## Example: Custom Training Call

```python
model, dataset = train_itur_model(
    train_csv="outputs/train_matched_data.csv",
    val_csv="outputs/val_matched_data.csv",
    test_csv="outputs/test_matched_data.csv",
    custom_channels=custom_channels,
    num_epochs=40,
    batch_size=16,
    learning_rate=0.0001,
    device='cuda',
)
```

## Outputs

- `outputs/best_itur_model.pth` — Best model weights
- `outputs/training_metrics.txt` — Training and validation metrics log
- `outputs/images/itur_training_curves.png` — Training/validation loss and R² plots
- `outputs/train_matched_data.csv`, `outputs/val_matched_data.csv`, `outputs/test_matched_data.csv` — Matched image/label info for each split

## Troubleshooting

- **SSL Certificate Errors**: If you see SSL errors when downloading pretrained weights, install Python certificates (see earlier in this README or ask for help).
- **Channel Mismatch**: Ensure the number of channels in `custom_channels` matches the model's `input_channels`.
- **No Images Matched**: Double-check your image filenames and CSV codes.

## License

MIT License

---

## About `ML_Updated.py`

`ML_Updated.py` is the core script of this repository and implements a robust, modular pipeline for urbanization prediction from Sentinel-2A satellite images. Below are its main features and architectural highlights:

### Main Features

- **Flexible Data Loading**

  - Supports both legacy (directory + CSV) and fast (pre-matched CSV) workflows.
  - Custom channel selection: Use any combination of Sentinel-2 bands and spectral indices (e.g., NDVI, NDBI, SAVI, MSAVI, NDWI, NMDI, NBR).
  - Handles normalization and resizing of images.

- **Deep Learning Model**

  - Uses a ResNet-based architecture, with support for TorchGeo/torchvision pretrained weights (Sentinel-2 or ImageNet).
  - Automatically adapts the input layer to match the number of input channels (bands + indices).
  - Custom regression head for ITUR value prediction.

- **Training Pipeline**

  - Data augmentation using `albumentations` for robust model generalization.
  - Early stopping: Monitors validation loss (or other metrics) and halts training if no improvement is seen for a configurable number of epochs. Restores the best model weights automatically.
  - Learning rate scheduling and optimizer configuration for stable training.
  - Automatic saving of best model and training metrics.

- **Evaluation & Visualization**

  - Computes and logs metrics: MSE, MAE, R² for validation and test sets.
  - Plots training/validation loss and R² curves for easy monitoring.
  - Saves all results and plots in the `outputs/` directory.

- **Robust Error Handling**

  - Handles missing or mismatched images/labels gracefully.
  - Provides clear error messages for common data issues (e.g., no images matched, missing columns).
  - Warns about non-finite values in predictions and metrics.

- **Modular Design**
  - All major steps (data loading, model definition, training, evaluation) are encapsulated in functions or classes for easy extension and reuse.
  - Easily customizable for new indices, bands, or model architectures.

### Early Stopping Details

- Early stopping is implemented via a dedicated `EarlyStopping` class.
- You can configure:
  - **Patience**: Number of epochs to wait for improvement before stopping.
  - **Minimum Delta**: Minimum change to qualify as an improvement.
  - **Restore Best Weights**: Automatically reloads the best model weights when stopping.
  - **Monitored Metric**: Can monitor validation loss, MAE, or R².
- Early stopping status and best epoch are logged and saved with the model.

### Example: Custom Channel Selection

```python
custom_channels = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12',
    'NDVI', 'NDBI', 'SAVI', 'MSAVI', 'NDWI', 'NMDI', 'NBR'
]
```

### Example: Training Call

```python
model, dataset = train_itur_model(
    train_csv="outputs/train_matched_data.csv",
    val_csv="outputs/val_matched_data.csv",
    test_csv="outputs/test_matched_data.csv",
    custom_channels=custom_channels,
    num_epochs=40,
    batch_size=16,
    learning_rate=0.0001,
    device='cuda',
)
```

---
