from monai import transforms
from math import pi

# Training
MODEL_NAME = "all_transforms"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
NUM_WORKERS = 8
SEED = 42
FAST_DEV_RUN = False
PATIENCE = 5

# DataModule
TRAINING_VENDOR = "A"
SPLIT_RATIO = 0.7
BATCH_SIZE = 8

# WANDB
PROJECT_NAME = "M-M"
ENTITY = "mahdi-imen"


spatial_keys = ("img", "seg")

intensity_keys = ("img",)

# # # Transforms
TRANSFORM = transforms.Compose(
    [
        transforms.SomeOf(
            [
                transforms.Rotated(keys=spatial_keys, angle=pi / 2, mode="nearest"),
                transforms.Rotated(keys=spatial_keys, angle=pi, mode="nearest"),
                transforms.RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=[1]),
                transforms.RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=[0]),
            ],
            num_transforms=[1, 4],
        ),
        transforms.RandZoomd(
            keys=spatial_keys,
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.8,
            mode=["bilinear", "nearest"],
        ),
        transforms.RandGaussianNoised(keys=intensity_keys, prob=0.5, mean=0, std=0.03),
        transforms.RandGaussianSmoothd(keys=intensity_keys, prob=0.3),
        transforms.Rand2DElasticd(
            keys=spatial_keys,
            prob=0.3,
            padding_mode="reflection",
            magnitude_range=(0, 3),
            spacing=45,
            mode=("bilinear", "nearest"),
        ),
        transforms.RandAffined(
            keys=spatial_keys,
            prob=0.3,
            rotate_range=0.3,
            translate_range=10,
            mode=("bilinear", "nearest"),
        ),
        transforms.RandAdjustContrastd(keys=intensity_keys, prob=0.5, gamma=(0.7, 1.4)),
        transforms.RandBiasFieldd(
            keys=intensity_keys, prob=0.5, coeff_range=(0.0, 0.3)
        ),
    ]
)
LOAD_TRANSFORM = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
