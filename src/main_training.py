'''
Run as:
train_model.py | model_architecture | 'notes about run for wandb'
'''
import os
import sys
import torch
import json
from torchvision import transforms
from tqdm import tqdm
import torch.utils.data as data_utils
import wandb
import pandas as pd
from models import initialise_models
import torchinfo
from data.data_loading import CustomDataset, split_data


def train_model(model, device, dataloaders, progress, criterion, optimizer, mode='tissueclass', num_epochs=25, scheduler=None):
    model = model.to(device)
    if mode not in ['tissueclass', 'her2status']:
        raise Exception("ERROR: model mode given not one of 'tissueclass' or 'her2status'.")
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
            
            # for inputs, labels, her2_labels in dataloaders[phase]:
            for inputs, labels in dataloaders[phase]:
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
                        # loss += (l2_lambda * l2_norm)
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
    num_epochs = 25
    
    model_name = str(sys.argv[1])
    
    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    SEED=42
    num_cpus=4
    
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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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
        ])
    }
    
    # using full set of data
    img_dir = '/home/21576262@su/masters/data/patches/'
    labels_dir = '/home/21576262@su/masters/data/labels/' 
    # img_dir = '/Volumes/AlexS/MastersData/processed/patches/'
    # labels_dir = '/Volumes/AlexS/MastersData/processed/labels/'

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.has_mps else "cpu") # run on mac
    # print(device)
    
    scheduler = None
    # Initialize the model for this run
    model, optimiser, criterion, parameters, scheduler = initialise_models.inceptionv3_preT(num_classes)
    # print("\n TORCHINFO SUMMARY \n")
    # print(torchinfo.summary(model, (3, 299, 299), batch_dim=0, col_names=('input_size', 'output_size', 'num_params', 'kernel_size'), verbose=0))
   
    # Initialize WandB  run
    wandb.login()
    parameters['batch_size'] = batch_size; parameters['epochs'] = num_epochs

    run = wandb.init(
        project="masters", # set project
        notes=sys.argv[2],
        config=parameters) # Track hyperparameters and run metadata
    
    # Save data split
    data_split = {'train': train_img_folders,
                  'val': val_img_folders,
                  'test': test_img_folders
                 }
    with open('/home/21576262@su/masters/models/data_splits/' + str(run.name) + '.json', 'w') as file:
        json.dump(data_split, file)
        
    progress = {'train': tqdm(total=len(dataloaders['train']), desc="Training progress"), 'val': tqdm(total=len(dataloaders['val']), desc="Validation progress")}

    # Train and evaluate
    # and send model to gpu
    model = train_model(model, device, dataloaders, progress, criterion, optimiser, num_epochs=num_epochs, scheduler=scheduler)
    
    # Save model
    torch.save(model.state_dict(), '/home/21576262@su/masters/models/' + str(run.name) + '_model_weights.pth')


if __name__ == '__main__':
    main()
