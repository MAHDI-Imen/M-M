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
- [ ] Rewrite training section and results visualiation as seperate modules for easier manipulation
- [x] Visualize results on a subject from each vendor from test set
- [ ] Rewrite the subject prediction per vendor section as a module

Notes
* Overfitting on training data



#### To do
- [ ] Read Unet paper
- [ ] Read DL techniques for automatic MRI segmentation paper
- [ ] Get familiar with the implemented code of Projet Ima206: halfway there
- [ ] Look up the medpy library and add evaluation to the pipeline 
- [ ] Apply augmentations
- [ ] Add Histogram and vendor partition to utils.visualization
- [ ] train.py
- [ ]



