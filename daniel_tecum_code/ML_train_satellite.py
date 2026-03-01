"""
ITUR Prediction Model for Quetzal Server Data
============================================

This module implements a machine learning pipeline to predict urbanization levels (ITUR values)
from Sentinel-2A satellite images stored on Quetzal server using TorchGeo.

Data Structure:
- Images: /mnt/data-r1/data/sentinel_images/BaseDatos_Sentinal2A/{CODE}.tif
- Labels: ITUR_resultados_nacional_v1.csv with COD_CELDA column
"""
##
import torch
import spyndex
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchgeo
from torchgeo.models import ResNet50_Weights, resnet152, ResNet152_Weights
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import warnings
from torchgeo.transforms import AppendNDVI, AppendNDBI
import csv
warnings.filterwarnings('ignore')
##

class QuetzalSentinelDataset(Dataset):
    """
    Custom dataset for loading Sentinel-2A images with flexible spectral indices
    and band selection using spyndex library for enhanced urbanization prediction.
    """

    # Sentinel-2 band mapping for spyndex (band names to indices)
    SENTINEL2_BANDS = {
        'B01': 0,  # Coastal aerosol
        'B02': 1,  # Blue
        'B03': 2,  # Green
        'B04': 3,  # Red
        'B05': 4,  # Red edge 1
        'B06': 5,  # Red edge 2
        'B07': 6,  # Red edge 3
        'B08': 7,  # NIR
        'B8A': 8,  # Red edge 4
        'B09': 9,  # Water vapor
        'B10': 10,  # SWIR - Cirrus
        'B11': 11,  # SWIR 1
        'B12': 12  # SWIR 2
    }

    # Reverse mapping for band names
    BAND_INDEX_TO_NAME = {v: k for k, v in SENTINEL2_BANDS.items()}

    def __init__(
            self,
            image_dir: str,
            csv_file: str,
            cod_celda_column: str = 'COD_CELDA',
            itur_column: str = 'ITUR',
            transform=None,
            image_size: Tuple[int, int] = (96, 96),
            max_samples: Optional[int] = 100,
            max_images_to_check: Optional[int] = None,
            # NEW: Flexible channel selection
            custom_channels: Optional[List[str]] = None,
            # DEPRECATED: Legacy parameters (kept for backward compatibility)
            use_all_bands: bool = None,
            use_spectral_indices: bool = None,
            spectral_indices: Optional[List[str]] = None,
            combine_bands_and_indices: bool = None,
            indices_only: bool = None
    ):
        """
        Initialize the Quetzal Sentinel dataset with flexible channel selection.

        Args:
            image_dir: Path to BaseDatos_Sentinel2A folder
            csv_file: Path to ITUR_resultados_nacional_v1.csv
            cod_celda_column: Column name for cell codes
            itur_column: Column name for ITUR values
            transform: Augmentation transforms
            image_size: Target image size
            max_samples: Limit number of final matched samples
            max_images_to_check: Limit number of image files to process
            custom_channels: List of band names and/or spectral indices to use
                           e.g., ['B04', 'B08', 'B11', 'B12', 'NDBI', 'NDVI', 'NDWI']

        Legacy parameters (deprecated but supported):
            use_all_bands: Whether to use all 13 bands (True) or just RGB (False)
            use_spectral_indices: Whether to compute spectral indices
            spectral_indices: List of spectral indices to compute
            combine_bands_and_indices: Whether to combine original bands with indices
            indices_only: Whether to use only spectral indices (no original bands)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size

        # Handle custom channels vs legacy parameters
        if custom_channels is not None:
            print("Using custom channel selection mode")
            self.custom_channels = custom_channels
            self.use_custom_channels = True
            # Separate bands and indices
            self.selected_bands = [ch for ch in custom_channels if ch in self.SENTINEL2_BANDS]
            self.selected_indices = [ch for ch in custom_channels if ch not in self.SENTINEL2_BANDS]
        else:
            print("Using legacy parameter mode")
            self.use_custom_channels = False
            # Handle legacy parameters
            self.use_all_bands = use_all_bands if use_all_bands is not None else True
            self.use_spectral_indices = use_spectral_indices if use_spectral_indices is not None else True
            self.combine_bands_and_indices = combine_bands_and_indices if combine_bands_and_indices is not None else True
            self.indices_only = indices_only if indices_only is not None else False

            # Set default spectral indices if none provided
            if spectral_indices is None:
                self.spectral_indices = [
                    'NDVI', 'NDBI', 'NDWI', 'SAVI', 'GNDVI', 'MNDWI', 'UI', 'IBI', 'BAEI', 'EBBI'
                ]
            else:
                self.spectral_indices = spectral_indices

        print(f"Loading ITUR data from: {csv_file}")

        if self.use_custom_channels:
            print(f"Custom channels: {self.custom_channels}")
            print(f"Selected bands: {self.selected_bands}")
            print(f"Selected indices: {self.selected_indices}")
        else:
            print(f"Using all 13 bands: {self.use_all_bands}")
            print(f"Using spectral indices: {self.use_spectral_indices}")
            if self.use_spectral_indices:
                print(f"Spectral indices to compute: {self.spectral_indices}")

        # Validate spectral indices
        if self.use_custom_channels:
            self._validate_custom_indices()
        elif self.use_spectral_indices:
            self._validate_spectral_indices()

        # Load ITUR CSV file
        try:
            self.itur_df = pd.read_csv(csv_file)
            print(f"Loaded CSV with {len(self.itur_df)} rows")
        except Exception as e:
            raise FileNotFoundError(f"Could not load CSV file: {e}")

        # Validate columns
        if cod_celda_column not in self.itur_df.columns:
            raise ValueError(f"Column '{cod_celda_column}' not found in CSV")

        # Find ITUR column
        itur_columns = [col for col in self.itur_df.columns if 'itur' in col.lower()]
        if itur_columns:
            itur_column = itur_columns[0]
            print(f"Using ITUR column: {itur_column}")
        elif itur_column not in self.itur_df.columns:
            print(f"Warning: '{itur_column}' not found")
            numeric_cols = self.itur_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                itur_column = numeric_cols[1]
                print(f"Using column '{itur_column}' as ITUR values")

        self.cod_celda_column = cod_celda_column
        self.itur_column = itur_column

        # Get list of available images
        print(f"Scanning for images in: {self.image_dir}")
        image_files = list(self.image_dir.glob("*.tif")) + list(self.image_dir.glob("*.tiff"))
        print(f"Found {len(image_files)} image files")

        # Limit images to process
        if max_images_to_check is not None and max_images_to_check < len(image_files):
            image_files = image_files[:max_images_to_check]
            print(
                f"Limited to first {len(image_files)} images for processing (max_images_to_check={max_images_to_check})")
        elif max_samples is not None and max_samples < len(image_files):
            limit = min(max_samples * 3, len(image_files))
            image_files = image_files[:limit]
            print(f"Limited to first {len(image_files)} images for processing (estimated for {max_samples} samples)")

        # Match images with ITUR data
        self.matched_data = []
        for img_path in tqdm(image_files, desc="Matching images"):
            code = img_path.stem
            matching_rows = self.itur_df[self.itur_df[cod_celda_column] == code]

            if len(matching_rows) == 1:
                itur_value = matching_rows[itur_column].iloc[0]
                if pd.notna(itur_value) and isinstance(itur_value, (int, float)):
                    self.matched_data.append({
                        'image_path': img_path,
                        'code': code,
                        'itur_value': float(itur_value)
                    })

                    if max_samples and len(self.matched_data) >= max_samples:
                        break

        print(f"Successfully matched {len(self.matched_data)} images")

        if len(self.matched_data) == 0:
            raise ValueError("No images could be matched with ITUR data")

        # Normalize ITUR values
        itur_values = [item['itur_value'] for item in self.matched_data]
        if max(itur_values) > 1.0:
            print(f"Normalizing ITUR values from [0, {max(itur_values):.2f}] to [0, 1]")
            self.itur_max = max(itur_values)
            for item in self.matched_data:
                item['itur_value'] = item['itur_value'] / self.itur_max
        else:
            self.itur_max = 1.0

        # Calculate final number of channels
        self.num_channels = self._calculate_num_channels()
        print(f"Final number of channels: {self.num_channels}")

    def _validate_custom_indices(self):
        """Validate that custom spectral indices are available in spyndex."""
        if not self.selected_indices:
            return

        available_indices = list(spyndex.indices.keys())
        invalid_indices = [idx for idx in self.selected_indices if idx not in available_indices]

        if invalid_indices:
            print(f"Warning: The following indices are not available: {invalid_indices}")
            print(f"Available indices: {len(available_indices)} total")

            # Remove invalid indices from custom channels
            valid_selected_indices = [idx for idx in self.selected_indices if idx in available_indices]
            self.selected_indices = valid_selected_indices

            # Update custom channels
            self.custom_channels = self.selected_bands + self.selected_indices
            print(f"Updated custom channels: {self.custom_channels}")

        print(f"Valid custom spectral indices: {len(self.selected_indices)}")

    def _validate_spectral_indices(self):
        """Validate that requested spectral indices are available in spyndex."""
        available_indices = list(spyndex.indices.keys())
        invalid_indices = [idx for idx in self.spectral_indices if idx not in available_indices]

        if invalid_indices:
            print(f"Warning: The following indices are not available: {invalid_indices}")
            print(f"Available indices: {len(available_indices)} total")
            print("Some popular indices for urbanization studies:")
            urbanization_indices = ['NDVI', 'NDBI', 'NDWI', 'SAVI', 'GNDVI', 'MNDWI', 'UI', 'IBI', 'BAEI', 'EBBI']
            for idx in urbanization_indices:
                if idx in available_indices:
                    print(f"  - {idx}: {spyndex.indices[idx].long_name}")

            # Remove invalid indices
            self.spectral_indices = [idx for idx in self.spectral_indices if idx in available_indices]

        print(f"Valid spectral indices to compute: {len(self.spectral_indices)}")

    def _calculate_num_channels(self):
        """Calculate the total number of channels based on configuration."""
        if self.use_custom_channels:
            return len(self.custom_channels)

        # Legacy calculation
        num_channels = 0

        if not self.indices_only:
            # Add original bands
            num_channels += 13 if self.use_all_bands else 3

        if self.use_spectral_indices:
            # Add spectral indices
            num_channels += len(self.spectral_indices)

        return num_channels

    def _extract_band_arrays(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract individual band arrays for spyndex computation."""
        bands = {}

        if image.shape[0] >= 13:
            # Map Sentinel-2 bands to spyndex parameter names
            bands.update({
                'B': image[1],  # Blue (B02)
                'G': image[2],  # Green (B03)
                'R': image[3],  # Red (B04)
                'RE1': image[4],  # Red edge 1 (B05)
                'RE2': image[5],  # Red edge 2 (B06)
                'RE3': image[6],  # Red edge 3 (B07)
                'N': image[7],  # NIR (B08)
                'RE4': image[8],  # Red edge 4 (B8A)
                'WV': image[9],  # Water vapor (B09)
                'S1': image[11],  # SWIR 1 (B11)
                'S2': image[12]  # SWIR 2 (B12)
            })
        else:
            # Fallback for RGB images
            bands.update({
                'B': image[0] if image.shape[0] > 0 else np.zeros_like(image[0]),
                'G': image[1] if image.shape[0] > 1 else np.zeros_like(image[0]),
                'R': image[2] if image.shape[0] > 2 else np.zeros_like(image[0]),
                'N': image[2] if image.shape[0] > 2 else np.zeros_like(image[0]),  # Use red as NIR fallback
                'S1': image[2] if image.shape[0] > 2 else np.zeros_like(image[0]),  # Use red as SWIR fallback
                'S2': image[2] if image.shape[0] > 2 else np.zeros_like(image[0])  # Use red as SWIR fallback
            })

        return bands

    def _compute_spectral_indices(self, bands: Dict[str, np.ndarray], indices_to_compute: List[str]) -> Dict[str, np.ndarray]:
        """Compute spectral indices using spyndex, with NaN/inf safety."""
        indices_data = {}

        for index_name in indices_to_compute:
            try:
                index_info = spyndex.indices[index_name]
                params = {}

                for band_param in index_info.bands:
                    if band_param in bands:
                        params[band_param] = bands[band_param]

                if hasattr(index_info, 'parameters'):
                    for param in index_info.parameters:
                        if param in spyndex.constants:
                            params[param] = spyndex.constants[param].default

                if 'L' not in params and 'L' in str(index_info.formula):
                    params['L'] = 0.5
                if 'C1' not in params and 'C1' in str(index_info.formula):
                    params['C1'] = 6.0
                if 'C2' not in params and 'C2' in str(index_info.formula):
                    params['C2'] = 7.5
                if 'gamma' not in params and 'gamma' in str(index_info.formula):
                    params['gamma'] = 1.0

                index_result = spyndex.computeIndex(index=index_name, params=params)

                if hasattr(index_result, 'sel'):
                    index_array = index_result.sel(index=index_name).values
                else:
                    index_array = np.array(index_result)

                # Sanitize values
                index_array = np.nan_to_num(index_array, nan=0.0, posinf=0.0, neginf=0.0)

                # Clip to valid reflectance range if needed
                index_array = np.clip(index_array, -1.0, 1.0)

                indices_data[index_name] = index_array

            except Exception as e:
                print(f"Warning: Could not compute index {index_name}: {e}")
                indices_data[index_name] = np.zeros_like(bands.get('R', np.zeros((96, 96))))

        return indices_data


    def _get_custom_channels_data(self, image: np.ndarray) -> np.ndarray:
        """Get data for custom selected channels."""
        channel_data = []

        # Extract band arrays for spectral indices computation
        bands_dict = self._extract_band_arrays(image)

        # Compute all required spectral indices
        indices_data = {}
        if self.selected_indices:
            indices_data = self._compute_spectral_indices(bands_dict, self.selected_indices)

        # Build channels in the order specified in custom_channels
        for channel in self.custom_channels:
            if channel in self.SENTINEL2_BANDS:
                # It's a band
                band_idx = self.SENTINEL2_BANDS[channel]
                if band_idx < image.shape[0]:
                    channel_data.append(image[band_idx])
                else:
                    # Band not available, use zeros
                    channel_data.append(np.zeros_like(image[0]))
            elif channel in indices_data:
                # It's a spectral index
                channel_data.append(indices_data[channel])
            else:
                # Unknown channel, use zeros
                print(f"Warning: Unknown channel {channel}, using zeros")
                channel_data.append(np.zeros_like(image[0]))

        return np.stack(channel_data, axis=0)

    def __len__(self):
        return len(self.matched_data)

    def __getitem__(self, idx):
        """Get a sample with flexible channel selection."""
        sample = self.matched_data[idx]
        image_path = sample['image_path']
        itur_value = sample['itur_value']

        try:
            # Load image
            with rasterio.open(image_path) as src:
                image = src.read()  # Shape: (bands, height, width)

            # Ensure we have the right number of bands
            if image.shape[0] < 13:
                # Pad with zeros or repeat bands
                while image.shape[0] < 13:
                    image = np.concatenate([image, image[-1:]], axis=0)
                image = image[:13]

            # Normalize pixel values
            image = image.astype(np.float32)
            if image.max() > 10000:
                image = image / 10000.0
            elif image.max() > 1.0:
                image = image / image.max()
            image = np.clip(image, 0, 1)

            # Get final image data based on mode
            if self.use_custom_channels:
                # Custom channel selection
                final_image = self._get_custom_channels_data(image)
            else:
                # Legacy mode
                final_bands = []

                # Add original bands if requested
                if not self.indices_only:
                    if self.use_all_bands:
                        final_bands.append(image)
                    else:
                        # Use only RGB bands
                        rgb_bands = image[[3, 2, 1]]  # Red, Green, Blue
                        final_bands.append(rgb_bands)

                # Add spectral indices if requested
                if self.use_spectral_indices:
                    bands_dict = self._extract_band_arrays(image)
                    indices_data = self._compute_spectral_indices(bands_dict, self.spectral_indices)

                    if indices_data:
                        indices_array = np.stack(list(indices_data.values()), axis=0)
                        final_bands.append(indices_array)

                # Combine all bands
                if final_bands:
                    final_image = np.concatenate(final_bands, axis=0)
                else:
                    final_image = image

            # Resize image to target size
            if final_image.shape[1:] != self.image_size:
                # Simple resize using numpy (you might want to use cv2 or PIL for better quality)
                from scipy.ndimage import zoom
                zoom_factors = (1, self.image_size[0] / final_image.shape[1], self.image_size[1] / final_image.shape[2])
                final_image = zoom(final_image, zoom_factors)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Create zero image with correct dimensions
            final_image = np.zeros((self.num_channels, self.image_size[0], self.image_size[1]), dtype=np.float32)

        # Transpose to (height, width, channels) for transforms
        final_image = np.transpose(final_image, (1, 2, 0))

        # Apply transforms
        if self.transform:
            try:
                augmented = self.transform(image=final_image)
                final_image = augmented['image']
            except Exception as e:
                print(f"Error applying transforms: {e}")
                final_image = torch.from_numpy(final_image).permute(2, 0, 1)
        else:
            final_image = torch.from_numpy(final_image).permute(2, 0, 1)

        return final_image, torch.tensor(itur_value, dtype=torch.float32)

    def get_sample_info(self, idx):
        """Get additional information about a sample."""
        return self.matched_data[idx]

    def save_matched_data(self, output_path: str):
        """Save the matched data to CSV for inspection."""
        df = pd.DataFrame(self.matched_data)
        df.to_csv(output_path, index=False)
        print(f"Saved matched data to: {output_path}")

    def get_channel_names(self) -> List[str]:
        """Get the names of all channels in the final image."""
        if self.use_custom_channels:
            return self.custom_channels.copy()

        # Legacy mode
        channel_names = []

        if not self.indices_only:
            if self.use_all_bands:
                band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                              'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
                channel_names.extend(band_names)
            else:
                channel_names.extend(['Red', 'Green', 'Blue'])

        if self.use_spectral_indices:
            channel_names.extend(self.spectral_indices)

        return channel_names

    def print_channel_configuration(self):
        """Print detailed information about the channel configuration."""
        print("=== CHANNEL CONFIGURATION ===")
        print(f"Total channels: {self.num_channels}")

        if self.use_custom_channels:
            print("Mode: Custom channel selection")
            print(f"Custom channels: {self.custom_channels}")
            print(f"Selected bands: {self.selected_bands}")
            print(f"Selected indices: {self.selected_indices}")
        else:
            print("Mode: Legacy parameters")
            print(f"Configuration:")
            print(f"  - use_all_bands: {self.use_all_bands}")
            print(f"  - use_spectral_indices: {self.use_spectral_indices}")
            print(f"  - indices_only: {self.indices_only}")
            print(f"  - combine_bands_and_indices: {self.combine_bands_and_indices}")

        channel_names = self.get_channel_names()
        print(f"\nChannel order:")
        for i, name in enumerate(channel_names):
            channel_type = "Band" if name in self.SENTINEL2_BANDS else "Index"
            print(f"  Channel {i:2d}: {name} ({channel_type})")

        # Show spectral indices details
        indices_to_show = self.selected_indices if self.use_custom_channels else self.spectral_indices
        if indices_to_show:
            print(f"\nSpectral indices details:")
            for idx in indices_to_show:
                if idx in spyndex.indices:
                    info = spyndex.indices[idx]
                    print(f"  - {idx}: {info.long_name}")
                    print(f"    Formula: {info.formula}")
                    print(f"    Required bands: {info.bands}")
                else:
                    print(f"  - {idx}: Not found in spyndex")


##
class ITURPredictor(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout: float = 0.5, input_channels: int = 13):
        super().__init__()

        if pretrained:
            try:
                weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
                self.backbone = resnet50(weights=weights)
                print("Using SENTINEL2_ALL_MOCO pretrained weights (ResNet50)")
            except:
                from torchvision.models import resnet50
                self.backbone = resnet50(pretrained=True)
                print("Using torchvision ResNet50 with ImageNet weights")
        else:
            self.backbone = resnet50(weights=None)

        # Replace input conv if not RGB
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Get feature dimensions
        num_features = self.backbone.fc.in_features

        # Custom regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

##
class EarlyStopping:
    """Early stopping implementation for PyTorch training loops."""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, mode='min', verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights when stopping
            mode (str): 'min' for loss (lower is better), 'max' for metrics like R² (higher is better)
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.verbose = verbose

        # Initialize tracking variables
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0

        # Set comparison operator based on mode
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - min_delta)
        else:  # mode == 'max'
            self.monitor_op = lambda current, best: current > (best + min_delta)

    def __call__(self, val_metric, model, epoch):
        """
        Check if training should stop.

        Args:
            val_metric: Current validation metric value
            model: PyTorch model
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = val_metric
            self.best_epoch = epoch
            self.save_checkpoint(model)
            if self.verbose:
                print(f"Initial best {self.mode} metric: {val_metric:.6f}")

        elif self.monitor_op(val_metric, self.best_score):
            # Improvement found
            self.best_score = val_metric
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(
                    f"New best {self.mode} metric: {val_metric:.6f} (improvement: {abs(val_metric - self.best_score):.6f})")

        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"Restored best weights from epoch {self.best_epoch}")
                return True

        return False

    def save_checkpoint(self, model):
        """Save the best model weights."""
        if self.restore_best_weights:
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}


##
def get_transforms(train: bool = True, image_size: Tuple[int, int] = (460, 460)):
    """Get data transforms for Sentinel-2A imagery without normalization."""
    if train:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2()
        ])
##

def explore_data(image_dir: str, csv_file: str, cod_celda_column: str = 'COD_CELDA'):
    """
    Explore the dataset to understand structure and identify issues.
    """
    print("=" * 60)
    print("DATASET EXPLORATION")

    # Check CSV file
    print(f"\n1. CSV File Analysis: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - COD_CELDA column exists: {cod_celda_column in df.columns}")

        if cod_celda_column in df.columns:
            print(f"   - Unique codes: {df[cod_celda_column].nunique()}")
            print(f"   - Sample codes: {df[cod_celda_column].head().tolist()}")

        # Look for ITUR-like columns
        itur_cols = [col for col in df.columns if 'itur' in col.lower() or 'urban' in col.lower()]
        print(f"   - Potential ITUR columns: {itur_cols}")

        # Show first few rows
        print(f"   - First 3 rows:")
        print(df.head(3))

    except Exception as e:
        print(f"   - Error reading CSV: {e}")

    # Check image directory
    print(f"\n2. Image Directory Analysis: {image_dir}")
    image_path = Path(image_dir)

    if image_path.exists():
        tif_files = list(image_path.glob("*.tif")) + list(image_path.glob("*.tiff"))
        print(f"   - Directory exists: Yes")
        print(f"   - TIFF files found: {len(tif_files)}")

        if tif_files:
            print(f"   - Sample filenames: {[f.name for f in tif_files[:5]]}")

            # Analyze a sample image
            sample_img = tif_files[0]
            try:
                with rasterio.open(sample_img) as src:
                    print(f"   - Sample image info:")
                    print(f"     * Shape: {src.shape}")
                    print(f"     * Bands: {src.count}")
                    print(f"     * Data type: {src.dtypes[0]}")
                    print(f"     * CRS: {src.crs}")

                    # Read a small sample
                    data = src.read(1, window=((0, 100), (0, 100)))
                    print(f"     * Value range: {data.min()} - {data.max()}")

            except Exception as e:
                print(f"   - Error reading sample image: {e}")
    else:
        print(f"   - Directory exists: No")

    print("\n" + "=" * 60)


##
def train_itur_model(
        image_dir: str = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A",
        csv_file: str = "ITUR_resultados_nacional_v1.csv",
        cod_celda_column: str = 'COD_CELDA',
        itur_column: str = 'RESUL_ITUR',
        num_epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.5,
        device: str = 'cuda',
        max_samples: Optional[int] = 100,
        max_images_to_check: Optional[int] = 300,  # NEW: Only check first 300 images
        # Early stopping parameters
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1e-6,
        monitor_metric: str = 'val_loss',  # 'val_loss', 'val_r2', 'val_mae'
        restore_best_weights: bool = True
):
    """Training pipeline with fast image processing limit."""

    print("Starting ITUR Model Training (Fast Mode)")
    print("=" * 50)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset with limited image checking
    print("\nCreating dataset...")
    custom_indices = ['B04', 'B08', 'B11', 'B12', 'NDBI', 'NBAI', 'NDVI', 'NDWI', 'UI', 'SAVI', 'B03', 'B02', 'B01']

    full_dataset = QuetzalSentinelDataset(
        image_dir=image_dir,
        csv_file=csv_file,
        cod_celda_column=cod_celda_column,
        itur_column=itur_column,
        transform=None,  # Transforms will be added later
        custom_channels=custom_indices,
        max_samples=max_samples,
        max_images_to_check=max_images_to_check,
        use_all_bands=True
    )

    # Split into train and validation indices
    print(f"\nSplitting dataset into train/validation ({1 - validation_split:.1%}/{validation_split:.1%})...")
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_split,
        random_state=42,
        shuffle=True
    )

    val_indices, test_indices = train_test_split(
        val_indices, test_size=0.6, random_state=42)  # 60% of temp = 30% of total

    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # Create transforms
    print("\nApplying transforms...")
    train_transforms = get_transforms(train=True, image_size= (90,90))
    val_transforms = get_transforms(train=False, image_size=(90,90))

    # Subset without transforms, then apply transforms with wrapper
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    # Apply transforms to subsets using TransformWrapper
    class TransformWrapper:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, target = self.dataset[idx]
            if self.transform:
                # Convert to numpy if it's a tensor
                if isinstance(image, torch.Tensor):
                    image = image.numpy().transpose(1, 2, 0)  # CHW -> HWC

                transformed = self.transform(image=image)
                image = transformed['image']

            return image, target

    # Wrap datasets with transforms
    train_dataset = TransformWrapper(train_subset, train_transforms)
    val_dataset = TransformWrapper(val_subset, val_transforms)
    test_dataset = TransformWrapper(test_subset, val_transforms)

    # Save matched data CSVs
    # Note: If you want to save matched data, you'll need to extract the `indices` from the original dataset:
    matched_df = pd.DataFrame(full_dataset.matched_data)
    matched_df.iloc[train_indices].to_csv('outputs/train_matched_data600000.csv', index=False)
    matched_df.iloc[val_indices].to_csv('outputs/val_matched_data600000.csv', index=False)
    matched_df.iloc[test_indices].to_csv('outputs/test_matched_data600000.csv', index=False)

    # Create data loaders with proper batch handling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Reduced to avoid potential issues
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  # FIXED: Drop last incomplete batch to avoid BatchNorm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False  # Keep all validation samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )

    # FIXED: Initialize model with correct input channels
    print("\nInitializing model...")
    input_channels = 13  # All Sentinel-2A bands
    model = ITURPredictor(pretrained=True, input_channels=input_channels)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    # Initialize early stopping
    mode = 'min' if monitor_metric in ['val_loss', 'val_mae'] else 'max'
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        restore_best_weights=restore_best_weights,
        mode=mode,
        verbose=True
    )

    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_r2_scores = []

    # Create output directories
    os.makedirs('outputs/images', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
            images, targets = images.to(device), targets.to(device)

            # FIXED: Skip batches with size 1 to avoid BatchNorm issues
            if images.size(0) == 1:
                continue

            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images, targets = images.to(device), targets.to(device)

                outputs = model(images).view(-1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_batches += 1

                # Collect predictions and targets
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / max(val_batches, 1)

        # Calculate metrics with NaN/Inf checking
        predictions = np.array(all_predictions)
        targets_array = np.array(all_targets)

        # FIXED: Check for NaN/Inf values and filter them out
        if len(predictions) > 0 and len(targets_array) > 0:
            # Create mask for finite values
            finite_mask = np.isfinite(predictions) & np.isfinite(targets_array)

            if np.sum(finite_mask) > 0:
                # Filter out non-finite values
                predictions_clean = predictions[finite_mask]
                targets_clean = targets_array[finite_mask]

                # Calculate metrics only on clean data
                mse = mean_squared_error(targets_clean, predictions_clean)
                mae = mean_absolute_error(targets_clean, predictions_clean)
                r2 = r2_score(targets_clean, predictions_clean)

                # Log warning if we had to filter out values
                if np.sum(~finite_mask) > 0:
                    print(f"Warning: Filtered out {np.sum(~finite_mask)} non-finite prediction values")
            else:
                # All values are non-finite
                print("Warning: All predictions are non-finite (NaN/Inf). Setting metrics to 0.")
                mse = mae = r2 = 0.0
        else:
            mse = mae = r2 = 0.0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_r2_scores.append(r2)

        scheduler.step(avg_val_loss)

        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f'Val MSE: {mse:.6f}')
        print(f'Val MAE: {mae:.6f}')
        print(f'Val R²: {r2:.6f}')

        # Determine metric to monitor for early stopping
        if monitor_metric == 'val_loss':
            monitored_value = avg_val_loss
        elif monitor_metric == 'val_r2':
            monitored_value = r2
        elif monitor_metric == 'val_mae':
            monitored_value = mae
        else:
            monitored_value = avg_val_loss  # default

        # Check early stopping
        if early_stopping(monitored_value, model, epoch + 1):
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"Best {monitor_metric}: {early_stopping.best_score:.6f} (epoch {early_stopping.best_epoch})")
            print(f"Training stopped {early_stopping_patience} epochs after last improvement")
            break

        # Save best model (traditional way, in addition to early stopping)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_r2': r2,
                'val_mse': mse,
                'val_mae': mae,
                'itur_max': full_dataset.itur_max,
                'input_channels': input_channels,
                'early_stopping_info': {
                    'patience': early_stopping_patience,
                    'monitor_metric': monitor_metric,
                    'best_epoch': early_stopping.best_epoch,
                    'stopped_early': False  # Will be updated if early stopping triggers
                }
            }, 'outputs/best_itur_model.pth')

        # Log metrics to file
        metrics_file = 'outputs/training_metrics.txt'
        with open(metrics_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"  Train Loss: {avg_train_loss:.6f}\n")
            f.write(f"  Val Loss: {avg_val_loss:.6f}\n")
            f.write(f"  Val MSE: {mse:.6f}\n")
            f.write(f"  Val MAE: {mae:.6f}\n")
            f.write(f"  Val R²: {r2:.6f}\n")
            f.write(f"  Early Stopping Counter: {early_stopping.counter}/{early_stopping_patience}\n")
            f.write("\n")

    else:
        # Training completed without early stopping
        print(f"\n✅ Training completed all {num_epochs} epochs without early stopping")

    # Update the saved model with early stopping info
    if os.path.exists('outputs/best_itur_model.pth'):
        checkpoint = torch.load('outputs/best_itur_model.pth')
        checkpoint['early_stopping_info']['stopped_early'] = early_stopping.counter >= early_stopping_patience
        torch.save(checkpoint, 'outputs/best_itur_model.pth')

    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_r2_scores)

    final_message = f"Training finished at epoch {epoch + 1}"
    if early_stopping.counter >= early_stopping_patience:
        final_message += f" (early stopping after {early_stopping_patience} epochs without improvement)"

    print(f'\n{final_message}')
    print(f'Best validation loss: {best_val_loss:.6f}')

    torch.save(model.state_dict(), 'outputs/best_itur_model.pth')

    #testing model performance
    # Load the best saved model
    print("\nEvaluating test performance with best model...")
    checkpoint = torch.load('outputs/best_itur_model.pth', map_location=device)

    # Create a fresh model instance and load the best weights
    eval_model = ITURPredictor(pretrained=True, input_channels=input_channels)
    eval_model.load_state_dict(torch.load('outputs/best_itur_model.pth', map_location=device))
    eval_model = eval_model.to(device)
    eval_model.eval()

    # Evaluate on TEST data (not training data)
    test_predictions = []
    test_targets = []

    print("Calculating R² for test data...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Test R² Evaluation"):
            images, targets = images.to(device), targets.to(device)

            outputs = eval_model(images).view(-1)

            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

    # Calculate test R² and other metrics with NaN/Inf checking
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    # FIXED: Apply the same NaN/Inf filtering for test data
    if len(test_predictions) > 0 and len(test_targets) > 0:
        # Create mask for finite values
        finite_mask = np.isfinite(test_predictions) & np.isfinite(test_targets)

        if np.sum(finite_mask) > 0:
            # Filter out non-finite values
            test_predictions_clean = test_predictions[finite_mask]
            test_targets_clean = test_targets[finite_mask]

            # Calculate metrics only on clean data
            test_r2 = r2_score(test_targets_clean, test_predictions_clean)
            test_mse = mean_squared_error(test_targets_clean, test_predictions_clean)
            test_mae = mean_absolute_error(test_targets_clean, test_predictions_clean)

            # Log warning if we had to filter out values
            if np.sum(~finite_mask) > 0:
                print(f"Warning: Filtered out {np.sum(~finite_mask)} non-finite test prediction values")
        else:
            # All values are non-finite
            print("Warning: All test predictions are non-finite (NaN/Inf). Setting metrics to 0.")
            test_r2 = test_mse = test_mae = 0.0

        print(f"Test R²: {test_r2:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test MAE: {test_mae:.6f}")

    # Save test metrics to training_metrics.txt file
    with open('outputs/training_metrics.txt', 'a') as f:
        f.write("="*50 + "\n")
        f.write("FINAL TEST PERFORMANCE (Best Model):\n")
        f.write("="*50 + "\n")
        f.write(f"Test R²: {test_r2:.6f}\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")

    return model, full_dataset


##
def plot_training_curves(train_losses, val_losses, val_r2_scores):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_r2_scores, label='Validation R²', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Validation R² Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/images/itur_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    print("ITUR Prediction - Quick Test")
    print("=" * 50)

    # Configuration for quick testing
    IMAGE_DIR = "/mnt/data-r1/data/sentinel_images/BaseDatos_Sentinel2A"
    CSV_FILE = "ITUR_resultados_nacional_v1.csv"

    # First, explore the data (this is fast)
    print("Step 1: Data Exploration")
    explore_data(IMAGE_DIR, CSV_FILE, 'COD_CELDA')

    print("\nStep 2: Quick Training Test")
    # Quick test with minimal resources

    model, dataset, = train_itur_model(
        image_dir=IMAGE_DIR,
        csv_file=CSV_FILE,
        cod_celda_column='COD_CELDA',
        itur_column='RESUL_ITUR',  # ← Correct column name from your data
        num_epochs=40,  # ← Just 5 epochs for quick test
        batch_size=16,  # ← 4, Small batch to fit in memory
        learning_rate=0.0001, # decreased for better results
        validation_split=0.5,
        device='cuda',  # ← Change to 'cpu' if no GPU
        max_samples = 1000,  # ← Only use 100 images total
        max_images_to_check = 1100,
    )

    print("\nQuick test completed!")
    print("Check the following files:")
    print("- best_itur_model.pth (saved model)")
    print("- matched_data.csv (dataset info)")
    print("- itur_training_curves.png (training plots)")
    print("Current working directory:", os.getcwd())
