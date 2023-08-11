from monai import transforms

# Training
MODEL_NAME = "benchmark"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_WORKERS = 8
SEED = 42
FAST_DEV_RUN = False
PATIENCE = 2

# DataModule
TRAINING_VENDOR = "A"
SPLIT_RATIO = 0.7
BATCH_SIZE = 8

# WANDB
PROJECT_NAME = "M-M"
ENTITY = "mahdi-imen"


# # # Transforms
TRANSFORM = None

LOAD_TRANSFORM = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
