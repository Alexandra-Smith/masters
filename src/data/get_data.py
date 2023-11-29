import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import torch.utils.data as data_utils
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Subset
from collections import Counter

class CustomDataset(Dataset):
    def __init__(self, img_folders, label_files, transform=None):
        self.img_folders = img_folders
        self.label_files = label_files
        self.transform = transform

        self.imgs = [] # Keeps image paths to load in the __getitem__ method
        self.labels = []
        self.cases = []

        # Load images and corresponding labels
        for i, (img_folder, label_file) in enumerate(zip(img_folders, label_files)):
            # print("Patch directory", img_folder, "\nLabel file", label_file)
            labels_pt = torch.load(label_file) # Load .pt file
            self.cases.append(img_folder.split('/')[-1])
            # Run through all patches from the case folder
            for i, img in enumerate(os.listdir(img_folder)):
                if os.path.isfile(os.path.join(img_folder, img)) and os.path.isfile(label_file):
                    # print(img_folder + img)
                    case_id = img_folder.split('/')[-1]
                    if img.startswith('._'):
                        img = img.replace('._', '')
                    idx = int(img.replace('.png', '').split("_")[1])
                    self.imgs.append(os.path.join(img_folder, img))
                    self.labels.append(labels_pt[idx].item()) # get label as int
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # Load image at given index
        image_path = self.imgs[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None: # Apply transformations
            image = self.transform(image)
        
        label = self.labels[idx] # Load corresponding image label

        return image, label # Return transformed image and label

# Split image folders into train, val, test
def split_all_data(patch_directory, split: list, seed):
    '''
    Function that takes in the split percentage for train/val/test sets, and randomly chooses which cases
    to allocate to which set (to ensure all patches from one case go into one set)
    Parameters:
        patch_directory: folder containing all patches
        split: list of integers for splitting sets
        seed: option to set the seed value for randomness
    Returns:
        3 lists for each of train/val/test, where each list contains the case names to be used in the set
    '''
    
    random.seed(seed)
    case_folders = os.listdir(patch_directory) # get 147 case folders
    
    d = {}
    for folder in case_folders:
        num_patches_in_folder = len(os.listdir(patch_directory + folder))
        d[folder] = num_patches_in_folder
    
    total_num_patches = sum(d.values())
    train_split, val_split, test_split = split
    train_num_patches = int((train_split/100)*total_num_patches)
    val_num_patches = int((val_split/100)*total_num_patches)

    # list all folders in the directory
    folders = [os.path.join(patch_directory, folder) for folder in os.listdir(patch_directory) if os.path.isdir(os.path.join(patch_directory, folder))]
    
    # SELECT TRAINING CASES
    train_cases = [] # store all selected cases
    num_selected_train = 0 # number of patches selected so far
    selected_folders = set() # a set to store the selected folder names to keep track of those already selected
    while num_selected_train < train_num_patches:
        folder = random.choice(folders)
        if folder not in selected_folders:
            case = folder.replace(patch_directory, '')
            num_patches = len(os.listdir(folder))
            num_selected_train += num_patches
            selected_folders.add(folder) # add to set of selected folders
            train_cases.append(case)

    # SELECT VAL CASES
    val_cases = [] # store all selected cases
    num_selected_val = 0 # number of patches selected so far
    while num_selected_val < val_num_patches:
        folder = random.choice(folders)
        if folder not in selected_folders:
            case = folder.replace(patch_directory, '')
            num_patches = len(os.listdir(folder))
            num_selected_val += num_patches
            selected_folders.add(folder)
            val_cases.append(case)

    # SELECT TEST CASES
    cases = [folder.replace(patch_directory, '') for folder in folders]
    used = train_cases+val_cases
    test_cases = [case for case in cases if case not in used]
    
    # test_patches = [len(os.listdir(patch_directory + folder)) for folder in test_cases]
    num_selected_test = sum([len(os.listdir(patch_directory + folder)) for folder in test_cases])
    # dict = {x: for x in ['train', 'val', 'test']}
    # print(f"Number of training patches: {num_selected_train} \nNumber of validation patches: {num_selected_val} \nNumber of test patches: {num_selected_test}")
    return train_cases, val_cases, test_cases


class HER2Dataset(Dataset):
    def __init__(self, img_folders, label_files, transform=None):
        self.img_folders = img_folders
        self.label_files = label_files
        self.transform = transform

        self.imgs = [] # Keeps image paths to load in the __getitem__ method
        self.status = []
        self.img_cases=[] # Image case corresponding for each patch
        self.cases = [] # Keep track of cases used in this set
        
        df_her2_status = get_her2_status_list()

        # Load images and corresponding labels
        for i, (img_folder, label_file) in enumerate(zip(img_folders, label_files)):
            labels_pt = torch.load(label_file) # Load .pt file
            self.cases.append(img_folder.split('/')[-1])
            # Run through all patches from the case folder
            for i, img in enumerate(os.listdir(img_folder)):
                if os.path.isfile(os.path.join(img_folder, img)) and os.path.isfile(label_file):
                    case_id = img_folder.split('/')[-1]
                    # self.img_cases.append(case_id)
                    if img.startswith('._'):
                        img = img.replace('._', '')
                    idx = int(img.replace('.png', '').split("_")[1])
                    if labels_pt[idx].item() == 1:
                        self.imgs.append(os.path.join(img_folder, img))
                        self.img_cases.append(case_id)
                        self.status.append(df_her2_status[case_id]) # get label as int
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # Load image at given index
        image_path = self.imgs[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None: # Apply transformations
            image = self.transform(image)
        
        her2_status = self.status[idx] # Load corresponding image her2 status
        case_id = self.img_cases[idx]

        return image, her2_status, case_id # Return transformed image and label and case_id for image

def get_her2_status_list():

    # file_path = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/HER2DataInfo.xlsx'
    file_path = '/home/21576262@su/masters/data/HER2DataInfo.xlsx'
    df = pd.read_excel(file_path)

    df.drop(df.index[-2:], inplace=True)

    df['Case ID'] = df['Case ID'].str.replace('TCGA-','')
    df['Case ID'] = df['Case ID'].str.replace('-01Z-00-DX1','')

    df['Clinical.HER2.status'] = df['Clinical.HER2.status'].map({'Negative': 0, 'Positive': 1}).astype(int)

    dictt = df.set_index('Case ID').to_dict()['Clinical.HER2.status']

    return dictt

def split_tumour_data(patch_directory, label_directory, split: list, seed):
    '''
    Given all labels, filter out the tumour tiles and then split them into train, val and test sets.
    '''
    
    random.seed(seed)
    case_files = os.listdir(label_directory) # get 147 label files
    
    d = {}
    
    for file in case_files:
        num_tumour_patches = 0
        labels_pt = torch.load(os.path.join(label_directory, file))
        for lbl in labels_pt:
            if lbl==1:
                num_tumour_patches+=1
        d[file.replace('.pt', '')] = num_tumour_patches

    total_num_tumour_patches = sum(d.values())
    train_split, val_split, test_split = split
    train_num_patches = int((train_split/100)*total_num_tumour_patches)
    val_num_patches = int((val_split/100)*total_num_tumour_patches)
    
    # list all folders in the directory
    folders = [os.path.join(patch_directory, folder) for folder in os.listdir(patch_directory) if os.path.isdir(os.path.join(patch_directory, folder))]
    
    # SELECT TRAINING CASES
    train_cases = [] # store all selected cases
    num_selected_train = 0 # number of patches selected so far
    selected_folders = set() # a set to store the selected folder names to keep track of those already selected
    while num_selected_train < train_num_patches:
        folder = random.choice(folders)
        if folder not in selected_folders:
            case = folder.replace(patch_directory, '')
            num_patches = d[case]
            # num_patches = len(os.listdir(folder))
            num_selected_train += num_patches
            selected_folders.add(folder) # add to set of selected folders
            train_cases.append(case)

    # SELECT VAL CASES
    val_cases = [] # store all selected cases
    num_selected_val = 0 # number of patches selected so far
    while num_selected_val < val_num_patches:
        folder = random.choice(folders)
        if folder not in selected_folders:
            case = folder.replace(patch_directory, '')
            # num_patches = len(os.listdir(folder))
            num_patches = d[case]
            num_selected_val += num_patches
            selected_folders.add(folder)
            val_cases.append(case)

    # SELECT TEST CASES
    cases = [folder.replace(patch_directory, '') for folder in folders]
    used = train_cases+val_cases
    test_cases = [case for case in cases if case not in used]
    
    # test_patches = [len(os.listdir(patch_directory + folder)) for folder in test_cases]
    num_selected_test = sum([d[case] for case in test_cases])
    # print(f"Total tumour patches: {total_num_tumour_patches} \nNumber of training patches: {num_selected_train} \nNumber of validation patches {num_selected_val} \nNumber of test patches {num_selected_test}")
    
    return train_cases, val_cases, test_cases

class RandomSpecificRotation:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        # Randomly decide whether to apply the rotation or not based on the probability
        if random.random() < self.probability:
            # Choose a random rotation angle from 90, 180, or 270 degrees
            angle = random.choice([90, 180, 270])

            # Apply the rotation to the image
            img = TF.rotate(img, angle)

        return img

def define_transforms(PATCH_SIZE, isResNet=False, isInception=False, isInceptionResnet=False):
    
    if isInception:
        INPUT_SIZE=299
    elif isResNet:
        INPUT_SIZE=224
    elif isInceptionResnet:
        INPUT_SIZE=299
    else:
        INPUT_SIZE=PATCH_SIZE
    
    # Initialise data transforms
    if isInception:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomSpecificRotation(),
                transforms.ColorJitter(brightness=0.25, contrast=[0.5, 1.75], saturation=[0.75, 1.25], hue=0.04),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #inception
            ]),
            'val': transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
            ]),
            'test' : transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
            ])
        }
    elif isInceptionResnet:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomSpecificRotation(),
                transforms.ColorJitter(brightness=0.25, contrast=[0.5, 1.75], saturation=[0.75, 1.25], hue=0.04),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #inceptionresnet
            ]),
            'val': transforms.Compose([
                transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # inceptionresnet
            ]),
            'test' : transforms.Compose([
                transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # inceptionresnet
            ])
        }
    else:
         data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomSpecificRotation(),
                transforms.ColorJitter(brightness=0.25, contrast=[0.5, 1.75], saturation=[0.75, 1.25], hue=0.04)
            ]),
            'val': transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor()
            ]),
            'test' : transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor()
            ])
        }
        
    return data_transforms

def get_seg_dataloaders(batch_size, SEED, Inception=False, InceptionResnet=False):
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=4
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)
    
    # using full set of data
    img_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 

    split=[70, 15, 15] # for splitting into train/val/test

    train_cases, val_cases, test_cases = split_all_data(img_dir, split, SEED)

    train_img_folders = [img_dir + case for case in train_cases]
    val_img_folders = [img_dir + case for case in val_cases]
    test_img_folders = [img_dir + case for case in test_cases]

    # Contains the file path for each .pt file for the cases used in each of the sets
    train_labels = [labels_dir + case + '.pt' for case in train_cases]
    val_labels = [labels_dir + case + '.pt' for case in val_cases]
    test_labels = [labels_dir + case + '.pt' for case in test_cases]

    image_datasets = {
        'train': CustomDataset(train_img_folders, train_labels, transform=data_transforms['train']),
        'val': CustomDataset(val_img_folders, val_labels, transform=data_transforms['val']),
        'test': CustomDataset(test_img_folders, test_labels, transform=data_transforms['test'])
    }
    # Create training, validation and test dataloaders
    dataloaders = {
        'train': data_utils.DataLoader(image_datasets['train'], batch_size=batch_size, num_workers=num_cpus, shuffle=True, drop_last=True),
        'val': data_utils.DataLoader(image_datasets['val'], batch_size=batch_size, num_workers=num_cpus, shuffle=True),
        'test': data_utils.DataLoader(image_datasets['test'], batch_size=batch_size, num_workers=num_cpus, shuffle=True)
    }
    
    print(f"Total patches: {len(dataloaders['train'])*batch_size + len(dataloaders['val'])*batch_size + len(dataloaders['test'])*batch_size} \nNumber of training patches: {len(dataloaders['train'])*batch_size} \nNumber of validation patches {len(dataloaders['val'])*batch_size} \nNumber of test patches {len(dataloaders['test'])*batch_size}")
    
    return train_cases, val_cases, test_cases, dataloaders

def split_her2(SEED):
    
    torch.manual_seed(SEED)
    # using full set of data
    img_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/'
    
    all_cases = os.listdir(img_dir)

    # Split the data into 70% training and 15% validation, 15% test
    train_cases, temp_data = train_test_split(all_cases, test_size=0.30, random_state=SEED)
    val_cases, test_cases = train_test_split(temp_data, test_size=0.50, random_state=SEED)
    
    return train_cases, val_cases, test_cases

def her2_dataloaders(batch_size, SEED, train_cases, val_cases, test_cases, Inception=False, InceptionResnet=False):
    
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
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)
    
    image_datasets = {
    'train': HER2Dataset(train_img_folders, train_labels, transform=data_transforms['train']),
    'val': HER2Dataset(val_img_folders, val_labels, transform=data_transforms['val']),
    'test': HER2Dataset(test_img_folders, test_labels, transform=data_transforms['test'])
    }
    
    labels = [label for _, label, _ in image_datasets["train"]]
    label_counts = Counter(labels)
        
    class_counts = [label_counts[0], label_counts[1]]
    
    # Compute class weights for balancing training dataset
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    weights = [class_weights[i] for i in labels]  # dataset_targets are your labels
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    # Create training, validation and test dataloaders
    dataloaders = {
        'train': data_utils.DataLoader(image_datasets['train'], sampler=weighted_sampler, batch_size=batch_size, num_workers=num_cpus),
        'val': data_utils.DataLoader(image_datasets['val'], batch_size=batch_size, num_workers=num_cpus, shuffle=True),
        'test': data_utils.DataLoader(image_datasets['test'], batch_size=batch_size, num_workers=num_cpus, shuffle=False)
    }
    
    print(f"Total tumour patches: {len(dataloaders['train'])*batch_size + len(dataloaders['val'])*batch_size + len(dataloaders['test'])*batch_size} \nNumber of training patches: {len(dataloaders['train'])*batch_size} \nNumber of validation patches {len(dataloaders['val'])*batch_size} \nNumber of test patches {len(dataloaders['test'])*batch_size}")
    
    return dataloaders

def get_train_dataloader(subfolders, batch_size, Inception=False, InceptionResnet=False):
    '''
    Given a list of subfolders, extract all patches in them and put into a train dataloader.
    (used for cross validation) - implements undersampling of majority class
    '''

    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=4
    main_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 
    img_folders = [main_dir + case for case in subfolders]
    label_files = [labels_dir + case + '.pt' for case in subfolders]

    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)
    dataset = CustomDataset(img_folders, label_files, transform=data_transforms['train'])
    
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
        
    class_counts = [label_counts[0], label_counts[1]]
    
    # OVERSAMPLE minority class
    # Compute class weights
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    weights = [class_weights[i] for i in labels]  # dataset_targets are your labels
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    # Create dataloader
    dataloader = data_utils.DataLoader(dataset, sampler=weighted_sampler, batch_size=batch_size, num_workers=num_cpus)
    
    return dataloader

def get_test_dataloader(subfolders, batch_size, Inception=False, InceptionResnet=False):
    '''
    Given list of subfolders, extract all patches in them and put into a test dataloader.
    '''
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=4
    main_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 
    test_img_folders = [main_dir + case for case in subfolders]
    test_labels = [labels_dir + case + '.pt' for case in subfolders]
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)

    test_dataset = CustomDataset(test_img_folders, test_labels, transform=data_transforms['test'])
    # Create dataloader
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_cpus, shuffle=True)
    return test_dataloader

def get_her2test_dataloader(subfolders, batch_size, Inception=False, InceptionResnet=False):
    '''
    Given list of subfolders, extract all patches in them and put into a test dataloader.
    '''
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=4
    main_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 
    test_img_folders = [main_dir + case for case in subfolders]
    test_labels = [labels_dir + case + '.pt' for case in subfolders]
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)

    test_dataset = HER2Dataset(test_img_folders, test_labels, transform=data_transforms['test'])
    # Create dataloader
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_cpus, shuffle=False, drop_last=False)
    return test_dataloader


def get_her2test_dataset(subfolders, batch_size, Inception=False, InceptionResnet=False):
    '''
    Given list of subfolders, extract all patches in them and put into a test dataloader.
    '''
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=4
    main_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 
    test_img_folders = [main_dir + case for case in subfolders]
    test_labels = [labels_dir + case + '.pt' for case in subfolders]
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)

    test_dataset = HER2Dataset(test_img_folders, test_labels, transform=data_transforms['test'])
    
    return test_dataset

def kfolds_split(k, seed):
    '''
    Given all data, split into k folds and return case names for each fold.
    '''
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)  # k-folds cross-validation
    
    main_dir = '/home/21576262@su/masters/data/patches/'
    case_folders = os.listdir(main_dir)
    
    # Store results
    splits = {}

    # Apply 5-fold cross-validation on the full dataset
    for fold, (train_val_index, test_index) in enumerate(kf.split(case_folders), 1):
        test_subjects = [case_folders[i] for i in test_index]
        train_subjects = [case_folders[i] for i in train_val_index]

        splits[f"Fold-{fold}"] = {
            'train': train_subjects,
            'test': test_subjects
        }

    # Print the results
    # for fold, data in splits.items():
    #     print(f"Fold-{fold}:\nTrain: {data['train']}\nTest: {data['test']}\n")
        
    return splits

def get_her2status_dataloaders(batch_size, SEED, Inception=False, InceptionResnet=False):
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    num_cpus=4
    
    data_transforms = define_transforms(PATCH_SIZE, isInception=Inception, isInceptionResnet=InceptionResnet)
    
    # using full set of data
    img_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 

    split=[70, 15, 15] # for splitting into train/val/test

    train_cases, val_cases, test_cases = split_tumour_data(img_dir, labels_dir, split, SEED)

    train_img_folders = [img_dir + case for case in train_cases]
    val_img_folders = [img_dir + case for case in val_cases]
    test_img_folders = [img_dir + case for case in test_cases]

    # Contains the file path for each .pt file for the cases used in each of the sets
    train_labels = [labels_dir + case + '.pt' for case in train_cases]
    val_labels = [labels_dir + case + '.pt' for case in val_cases]
    test_labels = [labels_dir + case + '.pt' for case in test_cases]

    image_datasets = {
        'train': HER2Dataset(train_img_folders, train_labels, transform=data_transforms['train']),
        'val': HER2Dataset(val_img_folders, val_labels, transform=data_transforms['val']),
        'test': HER2Dataset(test_img_folders, test_labels, transform=data_transforms['test'])
    }
    # Create training, validation and test dataloaders
    dataloaders = {
        'train': data_utils.DataLoader(image_datasets['train'], batch_size=batch_size, num_workers=num_cpus, shuffle=True, drop_last=True),
        'val': data_utils.DataLoader(image_datasets['val'], batch_size=batch_size, num_workers=num_cpus, shuffle=True),
        'test': data_utils.DataLoader(image_datasets['test'], batch_size=batch_size, num_workers=num_cpus, shuffle=True)
    }
    
    print(f"Total tumour patches: {len(dataloaders['train'])*batch_size + len(dataloaders['val'])*batch_size + len(dataloaders['test'])*batch_size} \nNumber of training patches: {len(dataloaders['train'])*batch_size} \nNumber of validation patches {len(dataloaders['val'])*batch_size} \nNumber of test patches {len(dataloaders['test'])*batch_size}")
    
    return train_cases, val_cases, test_cases, dataloaders