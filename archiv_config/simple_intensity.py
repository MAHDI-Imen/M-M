from monai import transforms
from math import pi

# Training
MODEL_NAME = "simple_intensity"
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
        transforms.RandGaussianNoised(
            keys=intensity_keys, prob=0.5, mean=0.0, std=0.03
        ),
        transforms.ScaleIntensityd(keys=intensity_keys, minv=0.0, maxv=1.0),
    ]
)
LOAD_TRANSFORM = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
