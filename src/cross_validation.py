'''
Run as:
train_model.py | 'notes about run for wandb'
'''
import os
import sys
import torch
import time
from tqdm import tqdm
import wandb
import json
import pandas as pd
from models import initialise_models
from data.get_data import kfolds_split, get_train_dataloader, get_test_dataloader


def train_model(model, device, dataloader, progress, criterion, optimizer, num_epochs=25, scheduler=None):
    # Training loop for only train set (no validation)
    model = model.to(device)
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        phase = 'train'

        progress.reset()
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            progress.update()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0] # extract tensor if output is a tuple
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':

                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)   
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        if phase == 'train':
            loss_train = epoch_loss
            acc_train = epoch_acc
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        print()
        
        # Apply learning rate scheduling if needed
        if scheduler != None:
            scheduler.step()
            
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": epoch,
            "Train Loss": loss_train,
            "Train Acc": acc_train
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
    
    model_name = 'inception'
    
    Inception = True if model_name == 'inception' else False
    InceptionResnet = True if model_name == 'inceptionresnet' else False
    
    print(f"Model name: {model_name}")
    
    SEED=42
    
    # CROSS VALIDATION
    k = 5
    data_splits = kfolds_split(k, SEED)
    fold_items = list(data_splits.items())
    
    for i in range(k):
        
        print(f"Fold {i}")
        
        # Get data
        fold, data = fold_items[i]
        train_cases = data['train']
        test_cases = data['test']
        start_time = time.time()
        train_dataloader = get_train_dataloader(train_cases, batch_size, Inception=Inception, InceptionResnet=InceptionResnet)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time elapsed: {elapsed_time} seconds.")
        
        # Detect if there is a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model, optimiser, criterion, parameters, scheduler = initialise_models.INCEPTIONv3(num_classes, checkpoint_path=None)
        # Initialize WandB run
        wandb.login()
        parameters['batch_size'] = batch_size; parameters['epochs'] = num_epochs
        
        run = wandb.init(
            project="masters", # set project
            group='stage1-CV',
            notes=sys.argv[1],
            config=parameters) # Track hyperparameters and run metadata
        # Save data split
        data_split = {'seed': SEED,
                      'train': train_cases,
                      'test': test_cases
                     }
        with open('/home/21576262@su/masters/models/data_splits/' + str(run.name) + '.json', 'w') as file:
            json.dump(data_split, file)

        progress = tqdm(total=len(train_dataloader), desc="Training progress")

        # Train and evaluate
        model = train_model(model, device, train_dataloader, progress, criterion, optimiser, num_epochs=num_epochs, scheduler=scheduler)

        # Save model
        torch.save(model.state_dict(), '/home/21576262@su/masters/models/' + str(run.name) + '_model_weights.pth')
        
        run.finish()

if __name__ == '__main__':
    main()
