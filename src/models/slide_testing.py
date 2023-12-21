import os
import sys
import torch
import json
import timm
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import metrics
from matplotlib.colors import LinearSegmentedColormap
sys.path.append('/home/21576262@su/masters/')
from src.data.get_data import get_her2test_dataloader, get_her2_status_list
from src.models.inception_model import InceptionV3
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
# from src.models import initialise_models


def main():
    
    num_classes = 2
    batch_size = 32
    cd, custom_grn = define_colours()

    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    SEED=0
    num_cpus=8
    
    model_name = 'inceptionresnet'
    if model_name == 'inception': 
        INPUT_SIZE=299
    else: INPUT_SIZE=PATCH_SIZE
    ResNet = True if model_name == 'resnet' else False
    Inception = True if model_name == 'inception' else False
    InceptionResnet = True if model_name == 'inceptionresnet' else False
    
    model_names = {'trim-valley-173': 'INCEPTIONv4'}
    model_names = {'spring-pyramid-177': 'RESNET34', 'trim-valley-173': 'INCEPTIONv4', 'drawn-serenity-176': 'INCEPTIONv3', 'dazzling-sea-175' : 'INCEPTIONRESNETv2'}
    
    results = []
    # Test each model
    for name in model_names.keys():
        print(f"Testing model: {name}")
        print(model_names[name])
        # Load model
        model_path = os.path.join('masters/models/', name + '_model_weights.pth')
        model = load_trained_models(num_classes, model_path, model_names[name])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device) # send model to GPU
        
        isInceptionResnet = True if model_names[name]=='INCEPTIONRESNETv2' else False
        isInception = True if (model_names[name]=='INCEPTIONv3' or model_names[name]=='INCEPTIONv4') else False

        # Get info on model data split
        json_path = os.path.join('masters/models/data_splits/', name + '.json')
        with open(json_path, 'r') as file:
            data = json.load(file)
        testing_folders = data['test']
        print(f"Total testing folders (slides) = {len(testing_folders)}")
        test_dataloader = get_her2test_dataloader(testing_folders, batch_size, Inception=Inception, InceptionResnet=InceptionResnet) # testing subset for this model (make sure this is the HER2 loader!!)

    #     true_labels = []
    #     case_ids = []
    #     for inputs, labels, cases in test_dataloader:
    #         true_labels.extend(labels.tolist())
    #         case_ids.extend(cases)


    #     df = pd.DataFrame({
    #         'case': case_ids,
    #         'label': true_labels
    #     })
    #     df.to_csv('/home/21576262@su/masters/test_data.csv', index=False)

        # Test model
        case_ids, true_labels, model_probabilities, model_predictions = test_model(model, test_dataloader, device)
        # Save model results
        model_results = {
            'model_name': name,
            'case_ids': case_ids,
            'true_labels': true_labels,
            'predicted_probs': model_probabilities,
            'predicted_classes': model_predictions
        }
        # save model results to json
        # ?
        # with open('/home/21576262@su/masters/results.json', 'w') as file:
        #         json.dump(model_results, file)

        # patch-level testing (AUC)
        # predicted_probs = [model_probabilities[i][1] for i in range(len(model_probabilities))]
        # auc_score = roc_auc_score(true_labels, predicted_probs)

        # slide-level testing (AUC)
        case_dict = test_slide_level(model_results)
        # print(f"Length of slide scores: {len(case_dict)}")

        true_labels = [case_dict[case]['true_class'] for case in case_dict]
        avg_probs = [case_dict[case]['slide_avg_prob'] for case in case_dict]
        positively_clasif = [case_dict[case]['%_classified_as_positive'] for case in case_dict]
        predictions = [round(case_dict[case]['slide_avg_prob']) for case in case_dict]

        # print(true_labels)
        # print(len(true_labels))
        # print(true_labels)
        # print(predictions)
        # print(avg_probs)
        # print(positively_clasif)
        # s = []
        # for p in positively_clasif:
        #     if p>0.5:
        #         s.append(1)
        #     else:
        #         s.append(0)
        # print(s)


        auc_score1 = roc_auc_score(true_labels, avg_probs)
        auc_score2 = roc_auc_score(true_labels, positively_clasif)

        print(auc_score1)
        print(auc_score2)
        
        slide_model_results = {
            'model_name': name,
            'true_labels': true_labels,
            'aggregated_slide_probs': avg_probs,
            '%_tiles_classified_positive': positively_clasif,
            'predicted_classes': predictions
        }
        results.append(slide_model_results)
        
    df = pd.DataFrame(results)
    # Save to JSON
    df.to_json('masters/reports/results/stage2_final_slide_results.json', orient='records')

#         # calculate roc curves
#         fpr1, tpr1, _ = roc_curve(true_labels, avg_probs)
#         fpr2, tpr2, _ = roc_curve(true_labels, positively_clasif)
#         # generate a no skill prediction (majority class)
#         ns_probs = [0 for _ in range(len(true_labels))]
#         ns_fpr, ns_tpr, _ = roc_curve(true_labels, ns_probs)
#         # plot the roc curve for the model
#         plt.plot(ns_fpr, ns_tpr, cd['green'], linestyle='--', label='No Skill')
#         plt.plot(fpr1, tpr1, cd['purple'], marker='.', label=f"Average predicted probability: {'{:.2f}'.format(auc_score1)}")
#         plt.plot(fpr2, tpr2, cd['yellow'], marker='.', label=f"Percentage postively classified: {'{:.2f}'.format(auc_score2)}")
#         # axis labels
#         plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
#         plt.legend()
#         # plt.title('AUC=%.3f' % (auc_score))
#         # plt.show()
#         plt.savefig("/home/21576262@su/masters/reports/results/" + name + '/roc_slide_level.png')
#         plt.clf()

#         cm = confusion_matrix(true_labels, predictions)
#         group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
#         group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
#         labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts,group_percentages)]
#         labels = np.asarray(labels).reshape(2,2)
#         sns.heatmap(cm, annot=labels, fmt='', cmap=custom_grn)
#         # plt.title('Confusion Matrix');
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#         # disp.plot(cmap='plasma', values_format='.0f')
#         # disp.plot(cmap='PuBuGn', values_format='.2%')
#         # plt.show()
#         plt.savefig("/home/21576262@su/masters/reports/results/" + name + '/cm_slide_level.png')
#         plt.clf()

#         report = metrics.classification_report(true_labels, predictions)
#         print(report)

def load_trained_models(num_classes, model_path, model_architecture):
    
    model_names = ['RESNET34', 'RESNET18', 'RESNET50', 'INCEPTIONv3', 'INCEPTIONv4', 'INCEPTIONRESNETv2']
    
    if model_architecture == 'RESNET34':
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 'RESNET18':
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 'RESNET50':
        model = models.resnet50()  
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture == 'INCEPTIONv3':
        model = InceptionV3(num_classes=num_classes)
    elif model_architecture == 'INCEPTIONv4':
        model = timm.create_model('inception_v4', pretrained=False, num_classes=num_classes) 
        model.classif = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(model.get_classifier().in_features, num_classes)
        )
    elif model_architecture == 'INCEPTIONRESNETv2':
        model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=num_classes)
        model.classif = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(model.get_classifier().in_features, num_classes)
        )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model 

def test_slide_level(model_results):
    case_ids = model_results['case_ids']
    predicted_probs = model_results['predicted_probs']
    predicted_classes = model_results['predicted_classes']
    true_labels = model_results['true_labels']
    
    # print(set(case_ids))
    l = []
    for item in case_ids:
        if item not in l:
            l.append(item)
    print(l)
    
    dicc = get_her2_status_list()
    # print(dicc)
    L = [dicc[case] for case in l]
    print(L)
 
    case_dict = defaultdict(lambda: {'probs': [], 'pred_classes': [], 'true_classes': []}) # default dictionary created

    for case, prob, pred_class, true_label in zip(case_ids, predicted_probs, predicted_classes, true_labels):
        case_dict[case]['probs'].append(prob[1])
        case_dict[case]['pred_classes'].append(pred_class)
        case_dict[case]['true_classes'].append(true_label)

    case_dict = dict(case_dict) # convert back to dict
    
    # Aggregate results per slide
    for case in case_dict:
        true_labels = case_dict[case]['true_classes']
        # Checking status
        if len(set(true_labels)) != 1:
            print(len(set(true_labels)))
            print(true_labels)
            raise ValueError(f"Tiles from case {case} are not all the same HER2 status ")
        # else:
        #     print(f"All tile share same status for case {case}")
        
        # check corresponds with given HER2 status
        # get dataframe with statuses
        # ?
        # first check then add to dict if corresponds else error
        case_dict[case]['true_class'] = set(true_labels).pop()
        
        # Aggregate results
        # --------------------
        # using average probability
        case_dict[case]['slide_avg_prob'] = np.mean(case_dict[case]['probs'])
        
        # using % positively classified tiles
        # num_correct_tiles = np.sum([case_dict[case]['true_classes'][i] == case_dict[case]['pred_classes'][i] for i in range(len(case_dict[case]['true_classes']))])
        num_tiles_classified_pos = np.sum([case_dict[case]['pred_classes'][i] == 1 for i in range(len(case_dict[case]['true_classes']))])
        total_slide_tiles = len(case_dict[case]['true_classes'])
        percentage_pos_classified = num_tiles_classified_pos/total_slide_tiles
        case_dict[case]['%_classified_as_positive'] = percentage_pos_classified
    
    return case_dict
        

def test_model(model, test_loader, device):
    
    correct = 0
    total = 0
    
    # Create a progress bar
    progress_bar = tqdm(test_loader, desc='Testing', unit='batch')

    with torch.no_grad():
        true_labels = []
        predictions = []
        probabilities = []
        case_ids = []
        for inputs, labels, cases in progress_bar:
            # move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels.extend(labels.tolist())
            case_ids.extend(cases)

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
    
    return case_ids, true_labels, probabilities, predictions
        
def load_trained_model(num_classes, model_path): 
    
    # model = InceptionV3(num_classes=num_classes)
    model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=num_classes) 
    model.classif = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(model.get_classifier().in_features, num_classes)
    )
    
    # model = models.resnet34()
    # # Modify the last layer for binary classification (output 2 classes)    
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the saved model state dict
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    
    return model

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
        

if __name__ == '__main__':
    main()