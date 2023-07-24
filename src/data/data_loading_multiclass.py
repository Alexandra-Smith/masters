# Create a custom PyTorch dataset to read in your images and apply transforms
import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, img_folders, label_files, transform=None):
        self.img_folders = img_folders
        self.label_files = label_files
        self.transform = transform

        self.imgs = [] # Keeps image paths to load in the __getitem__ method
        self.labels = []
        self.cases = [] # All cases in this set
        self.img_cases=[] # Image case for each patch
        self.HER2_labels = []

        df_her2_status = get_her2_status_list()

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
                    self.img_cases.append(case_id)
                    if img.startswith('._'):
                        img = img.replace('._', '')
                    idx = int(img.replace('.png', '').split("_")[1])
                    self.imgs.append(os.path.join(img_folder, img))
                    self.labels.append(labels_pt[idx].item()) # get label as int
                    if labels_pt[idx].item() == 1: # if tile is cancerous
                        self.HER2_labels.append(df_her2_status[case_id])
                    else: # if not tumorous, there is no HER2 label
                        self.HER2_labels.append(0)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # Load image at given index
        image_path = self.imgs[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None: # Apply transformations
            image = self.transform(image)
        
        label = self.labels[idx] # Load corresponding image label

        her2_label = self.HER2_labels[idx]
        
        return image, label, her2_label # Return transformed image and label

# Split image folders into train, val, test
def split_data(patch_directory, split: list, seed):
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
    print(f"Number of training patches: {num_selected_train} \nNumber of validation patches {num_selected_val} \nNumber of test patches {num_selected_test}")
    return train_cases, val_cases, test_cases

def get_her2_status_list():

    # file_path = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/HER2DataInfo.xlsx'
    file_path = '/home/21576262@su/masters/data/HER2DataInfo.xlsx'
    df = pd.read_excel(file_path)

    df.drop(df.index[-2:], inplace=True)

    df['Case ID'] = df['Case ID'].str.replace('TCGA-','')
    df['Case ID'] = df['Case ID'].str.replace('-01Z-00-DX1','')

    df['Clinical.HER2.status'] = df['Clinical.HER2.status'].map({'Negative': 1, 'Positive': 2}).astype(int)

    dictt = df.set_index('Case ID').to_dict()['Clinical.HER2.status']

    return dictt

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

def define_transforms(PATCH_SIZE, isResNet=False, isInception=False):
    
    if isInception:
        INPUT_SIZE=299
    elif isResNet:
        INPUT_SIZE=224
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