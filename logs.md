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
* Data will be loaded in memory all at once because it's not that big

#### Friday
- [x] Add validation loss to training loop
- [x] Add save model checkpoints
- [x] Calculate Dice-coeff
- [x] Rewrite training section and results visualiation as seperate modules for easier manipulation
- [x] Visualize results on a subject from each vendor from test set
- [x] Rewrite the subject prediction per vendor section as a module
- [x] train.py
- [x] Get familiar with the implemented code of Projet Ima206: General overview


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
- [x] Visualize a batch of images for a subject from every vendor before and after intensenty cropping
- [x] Train model on the two transforms (with and without percentile) and see results 
- [x] Use CrossEntropyLoss instead of BCEwithlogits
- [x] Meeting: Follow up
- [x] Try Padding 1 instead of True: same thing
- [x] Fix random seed problem
- [x] Early stopping



Notes:
* There are artifacts in the volumes that might make percentile results different for different vendors.
* Seed is not working correctly: slightly different results for two identical runs.
* upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation
* CrossEntropy is better

Meeting Notes:
* Evaluating 3D is different than evaluating 2D: check implementation on miseval
* ES and ED should be considered seperately

#### Wednesday
- [x] Evaluation for ed and es seperately
- [x] Move a copy of the original data set to the cluster
- [x] Check why there is an inconsistensy in the results
- [x] Fix bug in train when save best = True
- [x] Fine tune model params to get best results

Notes:
* The AHD was removed because it takes too much time to compute and it's not indicative in our case: voxel spacing is not uniform 
* Many hyperparams were tested: not a signficant change in the results
* Training is still non deterministic even if the conv is used for upsampling instead of linear 

#### Thursday
- [x] Add more padding
- [x] Crop (256, 256): out of memory error

Notes:
* Without padding results dropped significantly

#### Friday
- [x] Experiment with MonteCarlo dropout
- [x] Test various optimizers 
- [x] use black for auto formatting

Notes:
* Results are very similar despite all the variations tested

## Week 3
### Tasks
#### Monday
- [x] Fix github problem
- [x] Retrieve lost files
- [x] restructure code for result analysis 

#### Tuesday + Wednesday
- [x] Fix non deterministic problem
- [x] Make an automatic pipeline
    - [x] Decide on a config format: YAML or INI or TOML: TOML
    - [x] Design the config file structure
    - [x] write read_config funtion

#### Thursday
- [x] Create a new dataset class to be able to apply random augentation for each epoch
- [x] Improve data loading function:  change get_subjects_dir
- [x] Ensure dataloader is deterministic
- [x] Write a full pipeline from config file

Notes:
* Code became too complex and messy
* Using torchio only for ROI extraction and testtime augmentation, for training on 2D use monai


#### Friday
- [x] Branch out and restart project: branch name new_project
- [x] Use pytorch lightning
- [x] Learn Monai
- [x] Learn einops

- [x] Use https://einops.rocks/1-einops-basics/ for the reshape
- [x] Rerwite model class to incorporate pytorch lightning
- [x] Rewrite data loading functions and VendorDataset Class 

Notes:
* Preprocess metadata in advance: use two centers for A instead of naming a new vendor


## Week 4
### Tasks
#### Monday
- [x] write pre_process_metadata notebook: splitting by vendor
- [x] write pre_process_metadata notebook: gather stats for vendor metadata
- [x] Visualize distributions by vendor for x_dim, x_pixdim
- [ ] rewrite ROI extraction notebook
- [x] Readapat code to use centers instead of vendors
- [ ] Fix bug in training loop: one hot encoding 



#### To do
##### Reading
- [ ] Read Unet paper
- [ ] Read DL techniques for automatic MRI segmentation paper
- [ ] Studying Robustness of Semantic Segmentation under Domain Shift in cardiac MRI: Library batch generators
- [ ] Read paper miseval

##### Priorities
- [ ] Auto DocString Extension 
- [ ] Resampling uniform voxel spacing=1.25 : the whole image
- [ ] Verify the code for 3D in miseval: verify input format


##### Next steps
- [ ] Try training on 3D volume
- [ ] Apply augmentations : spatial((rotation(composed of big and small rotation), crop), flip, elastic): one at a time and then together. Intensity(Blur, Gaussian noise, gamma augmentation, Brightness) : Make this as a pipeline

- [ ] Test time augmentation

- [ ] Contrastive learning for automatic augmentation

##### If I have extra time
- [ ] Add Histogram and vendor partition to utils.visualization