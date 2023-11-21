'''
Run as:
train_model.py
'''
import os
import sys
import torch
from tqdm import tqdm
import wandb
import json
import pandas as pd
from models import initialise_models
from data.get_data import get_seg_dataloaders, split_her2, her2_dataloaders
from sklearn.metrics import f1_score


def train_model(model, device, dataloaders, progress, criterion, optimizer, num_epochs=25, scheduler=None):
    model = model.to(device)
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
            true_labels = []
            all_preds = []
            
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                progress[phase].update()
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0] # extract tensor if output is a tuple
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    true_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    
                    if phase == 'train':

                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)   
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                loss_train = epoch_loss
                acc_train = epoch_acc
            if phase == 'val':
                loss_valid = epoch_loss
                acc_valid = epoch_acc
                valid_f1 = f1_score(true_labels, all_preds)
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
            "Valid Acc": acc_valid,
            "Valid F1": valid_f1
        })      
            
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
    num_epochs = 50
    
    SEED=42
    train_cases, val_cases, test_cases = split_her2(SEED)
    
    # LOAD ALL MODELS
    # Define a list of model names (matching the function names in initialise_models)
    # model_names = ['RESNET34', 'RESNET18', 'INCEPTIONRESNETv2', 'INCEPTIONv4', 'INCEPTIONv3']
    model_names = ['INCEPTIONRESNETv2']
    model_containers = []
    for model_name in model_names:
        init_function = getattr(initialise_models, model_name)
        model_info = {}
        model, optimiser, criterion, parameters, scheduler = init_function(num_classes)
        model_info['model'] = model; model_info['optimiser'] = optimiser; model_info['criterion'] = criterion; model_info['parameters'] = parameters; model_info['scheduler'] = scheduler
        model_containers.append(model_info)
    
    for i, model_info in enumerate(model_containers):
        
        model = model_info['model']
        optimiser = model_info['optimiser']
        criterion = model_info['criterion']
        parameters = model_info['parameters']
        scheduler = model_info['scheduler']
        
        InceptionResnet = True if model_names[i]=='INCEPTIONRESNETv2' else False
        Inception = True if (model_names[i]=='INCEPTIONv3' or model_names[i]=='INCEPTIONv4') else False
        
        print(f"Model #{i}: {model_names[i]}")
        
        dataloaders = her2_dataloaders(batch_size, SEED, train_cases, val_cases, test_cases, Inception=Inception, InceptionResnet=InceptionResnet)
        
        # Detect if there is a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize WandB run
        wandb.login()
        parameters['batch_size'] = batch_size; parameters['epochs'] = num_epochs
        
        run = wandb.init(
            project="masters", # set project
            group='STAGE 2',
            # notes=sys.argv[1],
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
        model = train_model(model, device, dataloaders, progress, criterion, optimiser, num_epochs=num_epochs, scheduler=scheduler)

        # Save model
        torch.save(model.state_dict(), '/home/21576262@su/masters/models/' + str(run.name) + '_model_weights.pth')
        
        run.finish()
    
if __name__ == '__main__':
    main()
