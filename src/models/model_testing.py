'''
RUN SCRIPT AS:
model_testing.py path_to_model_weights
'''

import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import random
import torch.utils.data as data_utils
import wandb
from PIL import Image
import seaborn as sns
import sys
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
sys.path.append('/home/21576262@su/masters/')
from src.data.data_loading import CustomDataset, split_data, define_transforms
from src.models.inception_model import InceptionV3

def load_trained_model(num_classes, model_path): 

    # model = nn.Sequential(
    #         nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Flatten(),
    #         nn.Linear(64 * 32 * 32, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, num_classes),
    #     )

    # INCEPTION
    # Define model architecture
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    # # Replace last layers with new layers
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 2048),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.7),
    #     nn.Linear(2048, num_classes),
    #     nn.Softmax(dim=1)
    # )
    
    # RESNET
    # model = torchvision.models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = InceptionV3(num_classes=num_classes)
    
    # Load the saved model state dict
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    

    return model

def test_model(model, test_loader, device):
    
    correct = 0
    total = 0
    
    # Create a progress bar
    progress_bar = tqdm(test_loader, desc='Testing', unit='batch')

    with torch.no_grad():
        true_labels = []
        predictions = []
        probabilities = []
        for inputs, labels in progress_bar:
            # move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.extend(labels.tolist())

            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Update variables
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            probabilities.extend(probs.tolist())
            predictions.extend(predicted.tolist())

            # Update progress bar description
            progress_bar.set_postfix({'Accuracy': '{:.2f}%'.format((correct / total) * 100)})
    
    # # Compute accuracy (testing for now) --delete
    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))
    # Close the progress bar
    progress_bar.close()
    
    return true_labels, probabilities, predictions

def load_data(INPUT_SIZE, SEED, batch_size, num_cpus, ResNet, Inception):

    data_transforms = define_transforms(INPUT_SIZE, isResNet=ResNet, isInception=Inception)

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

    return dataloaders

def get_metrics(y_test, predictions):
    report = metrics.classification_report(y_test, predictions)
    return report

def roc_plot(y_test, model_probabilities, model_name):
    # keep probabilities for the positive outcome only
    predicted_probs = [model_probabilities[i][1] for i in range(len(model_probabilities))]
    # calculate scores
    auc_score = roc_auc_score(y_test, predicted_probs)
    # summarize scores
    print('ROC AUC=%.3f' % (auc_score))
    # calculate roc curves
    fpr, tpr, _ = roc_curve(y_test, predicted_probs)
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, color='mediumseagreen', linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, color='mediumorchid', marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('AUC=%.3f' % (auc_score))
    # plt.show()
    plt.savefig("/home/21576262@su/masters/reports/results/" + model_name + '/roc.png')
    plt.clf()

def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap='BuGn')
    plt.title('Confusion Matrix');
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap='plasma', values_format='.0f')
    # disp.plot(cmap='PuBuGn', values_format='.2%')
    # plt.show()
    fig_name = 'cm.png'
    plt.savefig("/home/21576262@su/masters/reports/results/" + model_name + '/' + fig_name)
    plt.clf()

def main():
    ##### SET PARAMETERS #####
    num_classes = 2
    batch_size = 32
    model_name = 'inception'

    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    SEED=42
    num_cpus=8
 
    if model_name == 'inception': 
        INPUT_SIZE=299
    elif model_name == 'resnet':
        INPUT_SIZE=224
    else: INPUT_SIZE=PATCH_SIZE
    
    ResNet = True if model_name == 'resnet' else False
    Inception = True if model_name == 'inception' else False

    # Load data
    dataloaders = load_data(INPUT_SIZE, SEED, batch_size, num_cpus, ResNet, Inception)

    # Load model
    model_path = sys.argv[1]
    model = load_trained_model(num_classes, model_path)

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model = model.to(device)

    true_labels, model_probabilities, model_predictions = test_model(model, dataloaders['test'], device)

    metrics = get_metrics(true_labels, model_predictions)
    print(metrics)
    
    name = model_path.split('/')[-1].split('_')[0]
    # Save predicted probabilities and predictions from model on test set
    D = {"true_labels": true_labels, "model_probabilities": model_probabilities, "model_class_preditions": model_predictions}
    with open("/home/21576262@su/masters/models/testing_data/" + name + '.json', 'w') as f:
        json.dump(D, f)
    
    my_dirc = '/home/21576262@su/masters/reports/results/' + name
    if not os.path.isdir(my_dirc):
        os.makedirs(my_dirc)   
        
    # Visualisation
    # to save figures
    roc_plot(true_labels, model_probabilities, name)
    plot_confusion_matrix(true_labels, model_predictions, name)

if __name__ == '__main__':
    main()