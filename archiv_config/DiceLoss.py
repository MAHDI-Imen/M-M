from monai import transforms
from math import pi

# Training
MODEL_NAME = "DiceLoss"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
NUM_WORKERS = 8
SEED = 42
FAST_DEV_RUN = False
PATIENCE = 8

# DataModule
TRAINING_VENDOR = "A"
SPLIT_RATIO = 0.7
BATCH_SIZE = 32

# WANDB
PROJECT_NAME = "M-M"
ENTITY = "mahdi-imen"


spatial_keys = ("img", "seg")

intensity_keys = ("img",)

# # # Transforms
TRANSFORM = transforms.Compose(
    [
        transforms.RandFlipd(keys=("img", "seg"), prob=0.5, spatial_axis=[1]),
        transforms.RandFlipd(keys=("img", "seg"), prob=0.5, spatial_axis=[0]),
        transforms.RandAffined(
            keys=["img", "seg"],
            padding_mode="constant",
            translate_range=(5, 5),
            mode=[5, "nearest"],
            prob=0.7,
        ),
        transforms.RandZoomd(
            keys=["img", "seg"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            mode="nearest-exact",
            padding_mode="constant",
        ),
        transforms.OneOf(
            [
                transforms.RandGibbsNoised(keys=("img",), prob=1, alpha=(0.0, 0.5)),
                transforms.RandGaussianSmoothd(
                    keys=("img"), prob=1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)
                ),
                transforms.RandBiasFieldd(
                    keys=("img",), prob=1, coeff_range=(0.1, 0.2)
                ),
            ]
        ),
    ]
)


# TRANSFORM = transforms.Compose(
#     [
#         transforms.OneOf(
#             [
#                 transforms.Compose(
#                     [
#                         transforms.SomeOf(
#                             [
#                                 transforms.Rotated(
#                                     keys=["img", "seg"], angle=pi / 2, mode="nearest"
#                                 ),
#                                 transforms.Rotated(
#                                     keys=["img", "seg"], angle=pi, mode="nearest"
#                                 ),
#                                 transforms.RandFlipd(
#                                     keys=("img", "seg"), prob=1, spatial_axis=[1]
#                                 ),
#                                 transforms.RandFlipd(
#                                     keys=("img", "seg"), prob=1, spatial_axis=[0]
#                                 ),
#                             ],
#                             num_transforms=[0, 4],
#                         ),
#                         transforms.RandAffined(
#                             keys=["img", "seg"],
#                             padding_mode="constant",
#                             translate_range=(20, 20),
#                             mode=[5, "nearest"],
#                             prob=1,
#                         ),
#                         transforms.RandZoomd(
#                             keys=["img", "seg"],
#                             min_zoom=1.0,
#                             max_zoom=1.1,
#                             prob=0.5,
#                             mode="nearest-exact",
#                         ),
#                     ]
#                 ),
#                 transforms.SomeOf(
#                     [
#                         transforms.RandHistogramShiftd(
#                             keys=("img",), prob=1, num_control_points=10
#                         ),
#                         transforms.SomeOf(
#                             [
#                                 transforms.RandGaussianNoised(
#                                     keys=("img",), prob=1, mean=0.0, std=0.05
#                                 ),
#                                 transforms.RandGaussianNoised(
#                                     keys=("img",), prob=1, mean=0.2, std=0.05
#                                 ),
#                                 # transforms.RandGaussianNoised(keys=("img",), prob=1, mean=0.8, std=0.1),
#                             ],
#                             num_transforms=[0, 1],
#                         ),
#                         transforms.RandShiftIntensityd(
#                             keys=("img",), prob=1, offsets=(0, 0.3)
#                         ),
#                         transforms.RandGaussianSmoothd(
#                             keys=("img",),
#                             prob=1,
#                             sigma_x=(0.3, 0.5),
#                             sigma_y=(0.3, 0.5),
#                         ),
#                         transforms.RandBiasFieldd(
#                             keys=("img",), prob=1, coeff_range=(0.1, 0.2)
#                         ),
#                     ],
#                     num_transforms=[0, 2],
#                 ),
#             ]
#         )
#     ]
# )


LOAD_TRANSFORM = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)
