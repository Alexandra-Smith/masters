import pickle
import torch
import sys
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
from data.preprocessing import *

SVS_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/svs_files/'
MASK_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/interim/masks/'

# Define variables
PATCH_SIZE=299
STRIDE=PATCH_SIZE ### non-overlapping patches

NUM_CLASSES=2

# Preprocess data
patches, labels = load_data(SVS_DIR, MASK_DIR, PATCH_SIZE, STRIDE, NUM_CLASSES)

# Save data
SAVE_DEST = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/processed/tests/'
torch.save(patches, SAVE_DEST + '00-patches.pt')
torch.save(labels, SAVE_DEST + '00-labels.pt')