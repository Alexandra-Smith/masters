import torch
import sys
import pandas as pd
import time
sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
from data.preprocessing import *

SVS_DIR='/Volumes/AlexS/MastersData/SVS files/'
MASK_DIR='/Volumes/AlexS/MastersData/QupathLabels/export10x/'

# only use svs files for which a valid mask was obtained
files = os.listdir(MASK_DIR)

# Get file codes (IDs)
file_codes = []
for file in files:
    if file.endswith('.DS_Store'):
        continue
    name = file.replace(MASK_DIR, '').replace('.png', '')
    file_codes.append(name)

# Define variables
PATCH_SIZE=256
STRIDE=PATCH_SIZE ### non-overlapping patches

NUM_CLASSES=2

# since = time.time()

# i = 1

# case_code = file_codes[i]
case_code = 'TCGA-D8-A1J9-01Z-00-DX1.F81FA9EF-8129-4E17-A9AD-2B850782CC18'
# Preprocess data
# patches, labels, df = load_indv_case(case_code, SVS_DIR, MASK_DIR, PATCH_SIZE, STRIDE, NUM_CLASSES)
patches = LOAD(case_code, SVS_DIR, MASK_DIR, PATCH_SIZE, STRIDE, NUM_CLASSES)

# Save data
# SAVE_DEST = '/Volumes/AlexS/MastersData/processed/'
# SAVE_DEST__ = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/processed/'
# torch.save(patches, SAVE_DEST + 'patches/' + case_code.split('.')[0] + '.pt')
# torch.save(labels, SAVE_DEST + 'labels/' + case_code.split('.')[0] + '.pt')

# # export data to csv file (do for first file)
# df.to_csv(SAVE_DEST__ + 'data_info.csv', float_format='%.8f')  # index=False means not to write row numbers

# read in csv file and append new data (rows)
# df.to_csv(SAVE_DEST__ + 'data_info.csv', mode='a', header=False, float_format='%.8f')


# time_elapsed = time.time() - since
# print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))