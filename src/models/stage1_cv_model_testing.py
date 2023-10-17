'''
RUN SCRIPT AS:
model_testing.py
'''

import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import random
import torch.utils.data as data_utils
import wandb
import timm
from PIL import Image
import seaborn as sns
import sys
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
sys.path.append('/home/21576262@su/masters/')
# from src.data.data_loading import CustomDataset, split_data, define_transforms
from src.data.get_data import get_seg_dataloaders, get_her2status_dataloaders, get_test_dataloader
from src.models.inception_model import InceptionV3

def load_trained_model(num_classes, model_path): 
    
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

def get_metrics(y_test, predictions):
    report = metrics.classification_report(y_test, predictions)
    return report

def roc_plot(y_test, model_probabilities, model_name, colours):
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
    plt.plot(ns_fpr, ns_tpr, colours['green'], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, colours['lightpurple'], marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('AUC=%.3f' % (auc_score))
    # plt.show()
    plt.savefig("/home/21576262@su/masters/reports/results/" + model_name + '/roc.png')
    plt.clf()

def plot_confusion_matrix(y_test, predictions, model_name, colourmap):
    cm = confusion_matrix(y_test, predictions)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm, annot=labels, fmt='', cmap=colourmap)
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

def main():
    ##### SET PARAMETERS #####
    num_classes = 2
    batch_size = 32
    model_name = 'inception'
    
    cd, custom_grn = define_colours()

    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    SEED=42
    num_cpus=8
 
    if model_name == 'inception': 
        INPUT_SIZE=299
    else: INPUT_SIZE=PATCH_SIZE
    ResNet = True if model_name == 'resnet' else False
    Inception = True if model_name == 'inception' else False
    InceptionResnet = True if model_name == 'inceptionresnet' else False
    
    # Testing on 5-fold models
    
    # Given list of models
    # mix params
    # model_names = ['glamorous-eon-111', 'gallant-elevator-110', 'valiant-wave-109', 'celestial-frog-108', 'sunny-lion-107']
    
    # Gamble params
    model_names = ['cool-cloud-122', 'cool-snow-121', 'worthy-valley-120', 'dandy-dew-119', 'valiant-frost-118']
    
    results = []
    
    # Test each model
    for name in model_names:
        print(f"Testing model: {name}")
        # Load model
        model_path = os.path.join('masters/models/', name + '_model_weights.pth')
        model = load_trained_model(num_classes, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device) # send model to GPU
        
        # Get info on model data split
        json_path = os.path.join('masters/models/data_splits/', name + '.json')
        with open(json_path, 'r') as file:
            data = json.load(file)
        testing_folders = data['test']
        test_dataloader = get_test_dataloader(testing_folders, batch_size, Inception=True, InceptionResnet=False) # testing subset for this model

        # Test model
        true_labels, model_probabilities, model_predictions = test_model(model, test_dataloader, device)
        # Save model results
        model_results = {
            'model_name': name,
            'true_labels': true_labels,
            'predicted_probs': model_probabilities,
            'predicted_classes': model_predictions
        }
        results.append(model_results)
        
        metrics = get_metrics(true_labels, model_predictions)
        print(metrics)
        
        my_dirc = '/home/21576262@su/masters/reports/results/' + name
        if not os.path.isdir(my_dirc):
            os.makedirs(my_dirc)
        
        # Visualisation
        # to save figures
        roc_plot(true_labels, model_probabilities, name, cd)
        plot_confusion_matrix(true_labels, model_predictions, name, custom_grn)
    

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save to JSON
    df.to_json('masters/reports/results/stage1_cv_results.json', orient='records')
       

if __name__ == '__main__':
    main()
