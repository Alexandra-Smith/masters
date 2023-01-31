from initialise_models import *
from train_model import *
from predict_model import *
from ..data.preprocessing import *


# Initialise new run on W&B
wandb.init(project="masters")

# Define data variables
NUM_CLASSES = 2
PATCH_SIZE = 299

# todo: load in the data
# Load data
files = 
file_codes = 
for file_name in files:
    # Get image and corresponding segmentation mask for each patient
    image = 
    mask = 
    # create patches for WSIs
    patches = image_to_patches()
    # create patches for segmentation masks
    mask_patches = image_to_patches()
# Get labels
# todo: finish this function
labels = get_patch_labels(mask_patches)

# todo: before model training, get example visual to confirm patches extracted correctly

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