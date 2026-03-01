"""
ITUR Prediction Model for Quetzal Server Data
============================================

This module implements a machine learning pipeline to predict urbanization levels (ITUR values)
from Sentinel-2A satellite images stored on Quetzal server using TorchGeo.

Data Structure:
- Images: /mnt/data-r1/data/sentinel_images/BaseDatos_Sentinal2A/{CODE}.tif
- Labels: ITUR_resultados_nacional_v1.csv with COD_CELDA column

Modular structure: All core utilities are imported from the train/ directory.
To switch between ResNet and ViT, set model_type in train_itur_model.
"""

import os
import warnings
warnings.filterwarnings('ignore')
from train.data_utils import QuetzalSentinelDataset, get_transforms, explore_data
from train.model_utils import ITURPredictor
from train.train_utils import train_itur_model, EarlyStopping, plot_training_curves

if __name__ == "__main__":
    print("ITUR Prediction - Quick Test")
    print("=" * 50)

    # Configuration for quick testing
    TRAIN_CSV = "train_matched_data600000.csv"
    VAL_CSV = "val_matched_data600000.csv"
    TEST_CSV = "test_matched_data600000.csv"

    # Specify which bands and indices to use (13 bands + indices)
    custom_channels = [
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12',
        'NDVI', 'NDBI', 'SAVI', 'MSAVI', 'NDWI', 'NMDI', 'NBR'
    ]

    print("Step 1: Skipping data exploration (using pre-matched CSVs)")

    print("\nStep 2: Training using pre-matched CSVs and custom channels (bands + indices)")
    # model, dataset = train_itur_model(
    #     train_csv=TRAIN_CSV,
    #     val_csv=VAL_CSV,
    #     test_csv=TEST_CSV,
    #     custom_channels=custom_channels,
    #     num_epochs=40,  # or as needed
    #     batch_size=16,
    #     learning_rate=0.0001,
    #     device='cuda',  # or 'cpu'
    #     validation_split=0.5,  # not used in this mode
    #     max_samples=0,  # not used in this mode
    #     max_images_to_check=0  # not used in this mode
    # )

    # To use ViT instead of ResNet, uncomment below:
    model, dataset = train_itur_model(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        custom_channels=custom_channels,
        num_epochs=40,
        batch_size=16,
        learning_rate=0.0001,
        device='cuda',
        validation_split=0.5,
        max_samples=0,
        max_images_to_check=0,
        model_type='vit',
        image_size = (224, 224),
        vit_model_name='vit_small_patch16_224'
    )

    print("\nQuick test completed!")
    print("Check the following files:")
    print("- best_itur_model.pth (saved model)")
    print("- matched_data.csv (dataset info)")
    print("- itur_training_curves.png (training plots)")
    print("Current working directory:", os.getcwd())
