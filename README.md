# M-M
M&amp;M Challenge - Internship Project

M&Ms Segmentation

Keywords: Deep Learning, segmentation, generalizability, multi-center, cardiac MRI, contrastive learning, domain randomization, adversarial loss

Context: Generalizability of Deep Learning (DL) algorithms, i.e. their ability to generalize beyond the development dataset, is an active research topic. The M&Ms challenge [1] was organized in October 2020 to test the generalizability of cardiac MRI segmentation algorithms in a multi-centre, multi-vendor and multi-disease setting. Winning methods used a variety of data augmentation techniques and test-time data augmentation.

Data: The M&Ms dataset [3] will be used for experiments. It consists of 345 3D Magnetic Resonance (MR) images of human hearts (left and right ventricles), with their associated ground truth pixel-wise labels (left and right ventricle cavities, and myocardium), as described here [1].

Objectives: The goal of this internship will be to revisit and expand on the results of the M&Ms challenge by answering several questions. What is the relative role of the various spatial [2] and intensity-based augmentations, and of test-time augmentation? Do results persist under a more stringent training/test split, with a single center seen at training time? Furthermore, we will investigate the impact of contrastive learning to complement data augmentation, based on the recently proposed epsilon-SupInfoNCE and FairKL losses. We will experiment both in the supervised setting (raw data and ground segmentations from a single center) and semi-supervised setting (raw data without ground truth segmentations from a second center). Depending on interest and time, we will compare these approaches with other domain randomization and domain adaptation techniques (adversarial losses). The underlying DL architecture will be the U-Net. 

Code: Experiments will be conducted using PyTorch and its ecosystem (for instance -but not necessarily- torchIO and MONAI).

[1] Campello, Victor M., et al. "Multi-centre, multi-vendor and multi-disease cardiac segmentation: the M&Ms challenge." IEEE Transactions on Medical Imaging 40.12 (2021): 3543-3554.
[2] Chen, Chen, et al. "Improving the generalizability of convolutional neural network-based segmentation on CMR images." Frontiers in cardiovascular medicine 7 (2020): 105. 
[3] https://www.ub.edu/mnms/
