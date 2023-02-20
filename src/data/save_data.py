import pickle
import torch
import sys
import time
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
from data.preprocessing import *

SVS_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/svs_files/'
MASK_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/interim/masks/'

# Define variables
PATCH_SIZE=299
STRIDE=PATCH_SIZE ### non-overlapping patches

NUM_CLASSES=2

since = time.time()

# Preprocess data
patches, labels = load_data(SVS_DIR, MASK_DIR, PATCH_SIZE, STRIDE, NUM_CLASSES)

# Save data
SAVE_DEST = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/processed/tests/'
torch.save(patches, SAVE_DEST + '02-patches.pt')
torch.save(labels, SAVE_DEST + '02-labels.pt')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))