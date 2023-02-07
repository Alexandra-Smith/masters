from initialise_models import *
from train_model import *
from predict_model import *
import sys
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
from data.preprocessing import *


# Initialise new run on W&B
wandb.init(project="masters")

# Define data variables
NUM_CLASSES = 2
PATCH_SIZE = 299

# todo: load in the data
# Load data
files = 
file_codes = 
all_patches = torch.empty((0, 3, PATCH_SIZE, PATCH_SIZE)) # initialise empty tensors to concatenate
all_gt_patches = torch.empty((0, PATCH_SIZE, PATCH_SIZE))
for file_name in files:
    # Get image and corresponding segmentation mask for each patient
    image = 
    mask = 
    # create patches for WSIs
    patches = image_to_patches()
    # create patches for segmentation masks
    mask_patches = image_to_patches()
    # get rid of background patches
    tissue_patches, gt_patches = discard_background_patches(patches, mask_patches, PATCH_SIZE)
    # concatenate all patches from all images together
    
    # * may have to use .unsqueeze() here as well
    all_patches = torch.cat((all_patches, tissue_patches), dim=0); all_gt_patches = torch.cat((all_gt_patches, gt_patches), dim=0)

# Get labels
# todo: finish this function
labels = get_patch_labels(all_gt_patches)

# Define model variable
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.9
# Define model
inception_v3_model, loss_func, optimiser = initialise_inceptionv3_model(NUM_CLASSES, LEARNING_RATE, MOMENTUM)

# Training
train_model(patches, labels, inception_v3_model, loss_func, optimiser, NUM_EPOCHS)

wandb.join()

# Make predictions