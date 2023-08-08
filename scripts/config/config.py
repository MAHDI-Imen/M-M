from monai import transforms

# Training
MODEL_NAME = "benchmark"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
NUM_WORKERS = 8
SEED = 42
FAST_DEV_RUN = False

# DataModule
TRAINING_VENDOR = "A"
SPLIT_RATIO = 0.7
BATCH_SIZE = 8

# WANDB
PROJECT_NAME = "M-M"
ENTITY = "imen-mahdi"


# # # Transforms
# TRANSFORMS = transforms.Compose(
#     [
#         transforms.RandAffined(
#             keys=["img", "seg"],
#             rotate_range=5,
#             padding_mode="zeros",
#             translate_range=10,
#             scale_range=0.1,
#             prob=1,
#         ),
#         transforms.RandZoomd(keys=["img", "seg"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
#         transforms.RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=(0, 1)),
#         transforms.Rand2DElasticd(
#             keys=["img", "seg"], prob=0.1, padding_mode="zeroes"
#         ),  # seperate
#     ]
# )


LOAD_TRANSFORM = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
