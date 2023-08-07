'''
Run as:
train_model.py | run_group | 'notes about run for wandb'
'''
import os
import sys
import torch
from tqdm import tqdm
import wandb
import json
import pandas as pd
from models import initialise_models
from data.get_data import get_seg_dataloaders, get_her2status_dataloaders


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
    
    torch.cuda.empty_cache()
    ##### SET PARAMETERS #####
    
    # Number of classes in the dataset
    num_classes = 2
    # Batch size for training
    batch_size = 32
    # Number of epochs to train for
    num_epochs = 25
    
    model_name = 'inception'
    
    InceptionResnet = True if model_name == 'inceptionresnet' else False
    Inception = True if model_name == 'inception' else False
    
    print(f"Model name: {model_name}")
    
    SEED=42
    
    # train_cases, val_cases, test_cases, dataloaders = get_seg_dataloaders(batch_size, SEED, Inception=Inception, InceptionResnet=InceptionResnet)
    train_cases, val_cases, test_cases, dataloaders = get_her2status_dataloaders(batch_size, SEED, Inception=Inception, InceptionResnet=InceptionResnet)

    # Detect if there is a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    
    # checkpoint = '/home/21576262@su/masters/models/earthy-field-65_model_weights.pth' # Gamble
    # Initialize the model for this run
    model, optimiser, criterion, parameters, scheduler = initialise_models.INCEPTIONv3(num_classes, checkpoint_path=None)
   
    # Initialize WandB run
    wandb.login()
    parameters['batch_size'] = batch_size; parameters['epochs'] = num_epochs

    run = wandb.init(
        project="masters", # set project
        group=str(sys.argv[1]),
        notes=sys.argv[2],
        config=parameters) # Track hyperparameters and run metadata
    
    # Save data split
    data_split = {'seed': SEED,
                  'train': train_cases,
                  'val': val_cases,
                  'test': test_cases
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
