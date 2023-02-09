from initialise_models import *
from train_model import *
from predict_model import *
import torch.utils.data as data
import sys
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
from data.preprocessing import *
from data.save_data import PATCH_SIZE, STRIDE, NUM_CLASSES, SAVE_DEST

# Initialise new run on W&B
wandb.init(project="masters")

# Load saved data
patches = torch.load(SAVE_DEST + "00-patches.pt")
labels = torch.laod(SAVE_DEST + "00-labels.pt")

# convert to PyTorch dataset
# dataset = data.TensorDataset(patches[0], patches[1])

# # convert to PyTorch data loader
# dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

# patches, labels = load_data(SVS_DIR, MASK_DIR, PATCH_SIZE, STRIDE, NUM_CLASSES)

# Define model variable
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# Define model
inception_v3_model, loss_func, optimiser = initialise_inceptionv3_model(NUM_CLASSES, LEARNING_RATE, MOMENTUM)

# Training
train_model(patches, labels, inception_v3_model, loss_func, optimiser, NUM_EPOCHS)

# Finalise the W&B run
wandb.join()

# save the trained model
torch.save(model.state_dict(), "trained_model.pt")

# Make predictions