# Weekly logs
## Week 1
### Tasks
#### Monday
- [x] Meeting 03/07: Initial set-up and Introduction
- [x] Read M&M paper
- [x] Download the dataset
- [x] Set-up git repo
- [x] Check two implementations of Unet provided

Meetings notes: 07/03:
* For intensity based augmentations, gamma is more interesting than linear transformations of contrast and brightness(reason: normalization)
* For segmentation evaluation metrics use Dice/HSD
* First task is to implement an initial and simple pipeline with unet with acceptable results
* Attention based methods can be later tested depending on progress

#### Tuesday
- [x] Load and visualize the data
- [x] Set up cluster settings and load data in cluster
- [x] Get familiar with TorchIO library
- [x] Extract ROI and save new files to save memory

Notes:
* Only kept ED and ES: indexes from metadata
* 3 files were not readable in validation and testing: 3 subjects discarded
* ROI extraction is done through a centered mask of the segmentation, some additional padding, resizing 
* New data takes around 1GB of memory storage compared to 15GB: one order of magnitude difference

#### Wednesday
- [x] Load new data
- [x] Visualize histogram of images from different vendors
- [x] Visualize distribution of vendors for each dataset (train, valid, test)
- [x] Try out different changes to data loader
- [x] Try to run the training loop for a UNET model on the data
- [x] Reread and understand how UNET is implemented  

Notes:
* Data loader has to be changed: we'll save each slice seperately
* Code must be split into seperate modules and only use the notebook for testing

#### Thursday
- [x] Restructure code into seperate modules
- [x] rewrite data loader function
- [x] Pytorch tutorial to understand DataLoader and Dataset classes
- [x] Modify data_loader to load slides as TensorDataset(not SubjectDataset) in case of 2D
- [x] write training loop
- [x] Test run for 100 epochs
- [x] Visualize results on Validation data

Notes:
Data will be loaded in memory all at once because it's not that big

#### Friday
- [x] Add validation loss to training loop
- [x] Add save model checkpoints
- [x] Calculate Dice-coeff
- [x] Rewrite training section and results visualiation as seperate modules for easier manipulation
- [x] Visualize results on a subject from each vendor from test set
- [x] Rewrite the subject prediction per vendor section as a module
- [x] train.py


Notes
* Overfitting on training data
* C8J7L5 was removed from valid
* E3F5U2, C8O0P2 were removed from test



## Week 2
### Tasks
#### Monday
- [x] Look up the medpy library and add evaluation to the pipeline
- [x] Use medeval instead of medpy
- [x] Load data by vendor
- [x] Train on only one vendor
- [x] Visualize results for each vendor seperately
- [x] Analyise results with a boxplot
- [x] Meeting: discuss results and next steps
- [x] Change scaling: add percentile


Notes:
* Medpy is outdated. Used medeval instead which is way better
* Voxel spacing is not the same for all images in the x and y directions


Meetings notes: 07/10:
* CrossEntropy instead of BCE

#### Tuesday
- [x] Results of A: results just on validation 
- [ ] https://einops.rocks/1-einops-basics/ for the reshape
- [ ] Try Padding 1 instead of True
- [ ] Early stopping
- [x] Visualize a slice before and after intensenty cropping

Notes:
* There are artifacts in the volumes that might make percentile results different for different vendors.

#### Wednesday

#### Thursday

#### Friday

#### To do
- [ ] Read Unet paper
- [ ] Read DL techniques for automatic MRI segmentation paper
- [ ] Studying Robustness of Semantic Segmentation under Domain Shift in cardiac MRI: Library batch generators
- [ ] Get familiar with the implemented code of Projet Ima206: halfway there

- [ ] Add Histogram and vendor partition to utils.visualization


- [ ] Use CrossEntropyLoss instead of BCEwithlogits
- [ ] Evaluation for ed and es seperately
- [ ] Verify the code for 3D in miseval: verify input format
- [ ] Add more padding
- [ ] Crop (256, 256)
- [ ] Resampling uniform voxel spacing=1.25 : the whole image

- [ ] Apply augmentations : spatial((rotation(composed of big and small rotation), crop), flip, elastic): one at a time and then together. Intensity(Blur, Gaussian noise, gamma augmentation, Brightness) : Make this as a pipeline

- [ ] Test time augmentation