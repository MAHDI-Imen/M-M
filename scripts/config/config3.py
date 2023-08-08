from monai.transforms import ScaleIntensity

# Training
MODEL_NAME = "Vendor_D"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_WORKERS = 8
SEED = 42
FAST_DEV_RUN = False

# DataModule
TRAINING_VENDOR = "D"
SPLIT_RATIO = 0.7
BATCH_SIZE = 8

# WANDB
PROJECT_NAME = "M-M"
ENTITY = "imen-mahdi"

# Transforms
LOAD_TRANSFORM = ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
