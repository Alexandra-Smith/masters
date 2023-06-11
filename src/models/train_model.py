'''
Run as:
train_model.py | model_architecture | 'notes about run for wandb'
'''
import os
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import random
import torch.utils.data as data_utils
from PIL import Image
import wandb
import pandas as pd
import initialise_models
import torchinfo

def train_model(model, mode, device, dataloaders, progress, criterion, optimizer, num_epochs=25, scheduler=None):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if mode not in ['tissueclass', 'her2status']:
        raise Exception("ERROR: model mode given not on one 'tissueclass' or 'her2status'.")
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            progress[phase].reset()
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels, her2_labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if mode == 'her2status':
                    her2_labels = her2_labels.to(device)
                
                progress[phase].update()
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0] # extract tensor if output is a tuple
                    _, preds = torch.max(outputs, 1)
                    if mode == 'tissueclass':
                        loss = criterion(outputs, labels)
                    if mode == 'her2status':
                        loss = criterion(outputs, her2_labels)
                    
                    if phase == 'train':
                        # L2 regularisation
                        # l2_lambda = 1e-4
                        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                        # loss = loss + l2_lambda * l2_norm
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                if mode == 'tissueclass':
                    running_corrects += torch.sum(preds == labels.data)
                if mode == 'her2status':
                    running_corrects += torch.sum(preds == her2_labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                loss_train = epoch_loss
                acc_train = epoch_acc
            if phase == 'val':
                loss_valid = epoch_loss
                acc_valid = epoch_acc
            print(f'Epoch {epoch + 1}/{num_epochs}, {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        print()
        
        # Apply learning rate scheduling if needed
        if scheduler != None:
            scheduler.step()
            
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": epoch,
            "Train Loss": loss_train,
            "Train Acc": acc_train,
            "Valid Loss": loss_valid,
            "Valid Acc": acc_valid})      
            
    return model

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

# Create a custom PyTorch dataset to read in your images and apply transforms
class CustomDataset(Dataset):
    def __init__(self, img_folders, label_files, transform=None):
        self.img_folders = img_folders
        self.label_files = label_files
        self.transform = transform

        self.imgs = [] # Keeps image paths to load in the __getitem__ method
        self.labels = []
        self.HER2_labels = []

        df_status = get_her2_status_labels()

        # Load images and corresponding labels
        for i, (img_folder, label_file) in enumerate(zip(img_folders, label_files)):
            # print("Patch directory", img_folder, "\nLabel file", label_file)
            labels_pt = torch.load(label_file) # Load .pt file
            # Run through all patches from the case folder
            for i, img in enumerate(os.listdir(img_folder)):
                if os.path.isfile(img_folder + '/' + img) and os.path.isfile(label_file):
                    # print(img_folder + img)
                    case_id = img_folder.split('/')[-1]
                    if img.startswith('._'):
                        img = img.replace('._', '')
                    idx = int(img.replace('.png', '').split("_")[1])
                    self.imgs.append(img_folder + '/' + img)
                    self.labels.append(labels_pt[idx].item()) # get label as int
                    if labels_pt[idx].item() == 1: # if tile is cancerous
                        self.HER2_labels.append(df_status[case_id])
                    else: # if not tumorous, there is no HER2 label
                        self.HER2_labels.append(None)
        
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

def get_her2_status_labels():

    file_path = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/raw/HER2DataInfo.xlsx'
    df = pd.read_excel(file_path)

    df.drop(df.index[-2:], inplace=True)

    df['Case ID'] = df['Case ID'].str.replace('TCGA-','')
    df['Case ID'] = df['Case ID'].str.replace('-01Z-00-DX1','')

    df['Clinical.HER2.status'] = df['Clinical.HER2.status'].map({'Negative': 0, 'Positive': 1}).astype(int)

    dict = df.set_index('Case ID').to_dict()['Clinical.HER2.status']

    return dict

# -------------------------------------------------------------------------------
# ----------------------------------MAIN METHOD----------------------------------
# -------------------------------------------------------------------------------

def main():
    ##### SET PARAMETERS #####
    # Number of classes in the dataset
    num_classes = 2
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    # Number of epochs to train for
    num_epochs = 20
    
    model_name = sys.argv[1]
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    SEED=42
    num_cpus=8
    
    if model_name == 'inception': 
        INPUT_SIZE=299
    elif model_name == 'resnet':
        INPUT_SIZE=224
    else: INPUT_SIZE=PATCH_SIZE

    # * can automate this
    # Initialise data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # inception
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
        ]),
    }
    
    # using full set of data
    # img_dir = '/home/21576262@su/masters/data/patches/'
    # labels_dir = '/home/21576262@su/masters/data/labels/' 
    img_dir = '/Volumes/AlexS/MastersData/processed/patches/'
    labels_dir = '/Volumes/AlexS/MastersData/processed/labels/'

    split=[70, 15, 15] # for splitting into train/val/test

    train_cases, val_cases, test_cases = split_data(img_dir, split, SEED)

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

    # Detect if we have a GPU available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.has_mps else "cpu") # run on mac
    print(device)
    
    scheduler = None
    # Initialize the model for this run
    model, optimiser, criterion, parameters, scheduler = initialise_models.INCEPTIONv3(num_classes)
    # print("\n TORCHINFO SUMMARY \n")
    # print(torchinfo.summary(model, (3, 299, 299), batch_dim=0, col_names=('input_size', 'output_size', 'num_params', 'kernel_size'), verbose=0))
   
    # Initialize WandB  run
    wandb.login()
    parameters['batch_size'] = batch_size; parameters['epochs'] = num_epochs

    run = wandb.init(
        # Set the project where this run will be logged
        project="masters",
        notes=sys.argv[2],
        # Track hyperparameters and run metadata
        config=parameters)
    progress = {'train': tqdm(total=len(dataloaders['train']), desc="Training progress"), 'val': tqdm(total=len(dataloaders['val']), desc="Validation progress")}

    # Train and evaluate
    # Send model to gpu
    model = train_model(model, 'tissueclass', device, dataloaders, progress, criterion, optimiser, num_epochs=num_epochs, scheduler=scheduler)
    
    # Save model
    torch.save(model.state_dict(), '../../models/' + str(run.name) + '_model_weights.pth')

if __name__ == '__main__':
    main()