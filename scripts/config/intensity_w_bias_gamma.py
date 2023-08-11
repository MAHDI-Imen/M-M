from monai import transforms

# Training
MODEL_NAME = "intensity_w_bias_gamma"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_WORKERS = 8
SEED = 42
FAST_DEV_RUN = False
PATIENCE = 3

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
            keys=intensity_keys, prob=0.9, mean=0, std=0.03),

        transforms.RandAdjustContrastd(
            keys=intensity_keys, prob=0.8, gamma=(0.7, 1.4)),

        transforms.RandBiasFieldd(
            keys=intensity_keys, prob=0.8, coeff_range=(0., 0.3)),

        
    ]
)

LOAD_TRANSFORM = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
