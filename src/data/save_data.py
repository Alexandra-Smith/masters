import pickle
import torch
import sys
import time
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
from data.preprocessing import *

SVS_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/svs_files/'
MASK_DIR='/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/interim/masks/'

svs_files = os.listdir(SVS_DIR)

# Get file codes (IDs)
file_codes = []
for file in svs_files:
    if file.endswith('.DS_Store'):
        continue
    name = file.replace(SVS_DIR, '').replace('.svs', '')
    file_codes.append(name)

# Define variables
PATCH_SIZE=299
STRIDE=PATCH_SIZE ### non-overlapping patches

NUM_CLASSES=2

# since = time.time()

i = 0

case = file_codes[i]
# Preprocess data
patches, labels = load_indv_case(case, SVS_DIR, MASK_DIR, PATCH_SIZE, STRIDE, NUM_CLASSES)

# Save data
SAVE_DEST = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/processed/'
torch.save(patches, SAVE_DEST + 'patches/' + case + '.pt')
torch.save(labels, SAVE_DEST + 'labels/' + case + '.pt')

# time_elapsed = time.time() - since
# print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))