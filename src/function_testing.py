import torch
import numpy as np
import sys 
sys.path.append('/home/21576262@su/masters/src')
from data.get_data import get_seg_dataloaders, her2_dataloaders
from data.data_loading import CustomDataset, define_transforms, split_data
from data.get_data import split_tumour_data, HER2Dataset, get_her2_status_list 
from data.get_data import split_her2, her2_dataloaders
from models import initialise_models
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap

def main():
    
    # Number of classes in the dataset
    num_classes = 2
    # Batch size for training
    batch_size = 32
    # Number of epochs to train for
    num_epochs = 50
    
    SEED=42
    train_cases, val_cases, test_cases = split_her2(SEED)
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=8
    
    img_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/'
    
    train_img_folders = [img_dir + case for case in train_cases]
    val_img_folders = [img_dir + case for case in val_cases]
    test_img_folders = [img_dir + case for case in test_cases]

    # Contains the file path for each .pt file for the cases used in each of the sets
    train_labels = [labels_dir + case + '.pt' for case in train_cases]
    val_labels = [labels_dir + case + '.pt' for case in val_cases]
    test_labels = [labels_dir + case + '.pt' for case in test_cases]
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=False, isInceptionResnet=False)
    
    image_datasets = {
    'train': HER2Dataset(train_img_folders, train_labels, transform=data_transforms['train']),
    'val': HER2Dataset(val_img_folders, val_labels, transform=data_transforms['val']),
    'test': HER2Dataset(test_img_folders, test_labels, transform=data_transforms['test'])
    }
    
    labels_val = [label for _, label, _ in image_datasets['val']]
    label_counts_val = Counter(labels_val)
    class_counts_val = [label_counts_val[0], label_counts_val[1]]
    print(f"Validation counts {class_counts_val}")
    
    labels_test = [label for _, label, _ in image_datasets['test']]
    label_counts_test = Counter(labels_test)
    class_counts_test = [label_counts_test[0], label_counts_test[1]]
    print(f"Test counts {class_counts_test}")
    
        
#     train_cases, val_cases, test_cases = split_her2(SEED=42)
    
#     cd, custom_grn = define_colours()

#     img_dir = '/home/21576262@su/masters/data/patches/'
#     labels_dir = '/home/21576262@su/masters/data/labels/'

#     train_img_folders = [img_dir + case for case in train_cases]
#     val_img_folders = [img_dir + case for case in val_cases]
#     test_img_folders = [img_dir + case for case in test_cases]

#     # Contains the file path for each .pt file for the cases used in each of the sets
#     train_labels = [labels_dir + case + '.pt' for case in train_cases]
#     val_labels = [labels_dir + case + '.pt' for case in val_cases]
#     test_labels = [labels_dir + case + '.pt' for case in test_cases]

#     PATCH_SIZE=256
#     STRIDE=PATCH_SIZE
#     num_cpus=4
#     batch_size=32

#     data_transforms = define_transforms(PATCH_SIZE, isInception=True, isInceptionResnet=False)

#     dataset = HER2Dataset(train_img_folders, train_labels, transform=data_transforms['train'])

#     labels = [label for _, label, _ in dataset]
#     label_counts = Counter(labels)

#     class_counts = [label_counts[0], label_counts[1]]
    
#     print(class_counts)

#     # Compute class weights
#     class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
#     weights = [class_weights[i] for i in labels]  # dataset_targets are your labels
#     weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

#     # Create dataloader
#     dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, num_workers=num_cpus)
#     weighted_dataloader = data_utils.DataLoader(dataset, sampler=weighted_sampler, batch_size=batch_size, num_workers=num_cpus)

#     class_0_batch_counts, class_1_batch_counts, ids_seen = visualise_dataloader(dataloader, "original.jpg", {0: "HER2-", 1: "HER2+"})

#     class_0_batch_counts, class_1_batch_counts, ids_seen = visualise_dataloader(weighted_dataloader, "weighted.jpg", {0: "HER2-", 1: "HER2+"})

    
def define_colours():
    M_darkpurple = '#783CBB'
    M_lightpurple = '#A385DB'
    # M_green = '#479C8A'
    M_green = '#0a888a'
    M_yellow = '#FFDD99'
    M_lightpink = '#EFA9CD'
    M_darkpink = '#E953AD'

    colour_list = [M_lightpink, M_green, M_darkpurple, M_darkpink, M_lightpurple, M_yellow]
    cd = {'lightpink': M_lightpink, 'lightpurple': M_lightpurple, 'green': M_green, 'purple': M_darkpurple, 'pink': M_darkpink, 'yellow': M_yellow}
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colour_list)
    
    # Create custom gradient colourmap
    rgb_tuples =[(255, 255, 255),(249, 251, 251),(243, 247, 246),(237, 242, 242),(231, 238, 238),
                 (225, 234, 234),(219, 230, 230),(213, 226, 225),(207, 221, 221),(202, 217, 217),
                 (196, 213, 213),(190, 209, 209),(184, 205, 205),(178, 201, 201),(172, 197, 196),
                 (167, 193, 192),(161, 188, 188),(155, 184, 184),(149, 180, 180),(144, 176, 176),
                 (138, 172, 172),(132, 168, 168),(126, 164, 164),(121, 160, 160),(115, 157, 156),
                 (109, 153, 153),(103, 149, 149),(98, 145, 145),(92, 141, 141),(86, 137, 137),
                 (80, 133, 133),(74, 129, 130),(68, 125, 126),(61, 122, 122),(55, 118, 118),
                 (48, 114, 115),(41, 110, 111),(33, 106, 107),(24, 103, 104),(11, 99, 100)]
    # Normalize RGB color values to the range [0, 1]
    normalised_colours = [[r / 255, g / 255, b / 255] for r, g, b in rgb_tuples]
    custom_grn = LinearSegmentedColormap.from_list('Grn', normalised_colours, N=len(normalised_colours))
    
    return cd, custom_grn

def visualise_dataloader(dl, fig_name, id_to_label=None, with_outputs=True):
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []
    
    cd, custom_grn = define_colours()

    for i, batch in enumerate(dl):

        idxs = batch[0][:, 0].tolist()
        classes = batch[1]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        # idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        else:
            raise ValueError("More than two classes detected")
        
    # print(class_0_batch_counts)

    if with_outputs:
        fig, ax = plt.subplots(1, figsize=(15, 15))

        ind = np.arange(len(class_0_batch_counts))
        width = 0.35

        ax.bar(
            ind,
            class_0_batch_counts,
            width,
            color=cd['green'],
            label=(id_to_label[0] if id_to_label is not None else "0")
        )
        ax.bar(
            ind + width,
            class_1_batch_counts,
            width,
            color=cd['pink'],
            label=(id_to_label[1] if id_to_label is not None else "1")
        )
        ax.set_xticks(ind, ind + 1)
        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of images in batch", fontsize=12)
        ax.set_aspect("equal")

        plt.legend()
        # plt.show()
        plt.savefig(fig_name)

        # num_images_seen = len(idxs_seen)

        print(
            f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 32).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 32).mean()}'
        )
        print("=============")
        # print(f"Num. unique images seen: {len(set(idxs_seen))}/{total_num_images}")
        print(np.sum(class_0_batch_counts))
        print(np.sum(class_1_batch_counts))
        print("=============")
    return class_0_batch_counts, class_1_batch_counts, 0

if __name__ == '__main__':
    main()