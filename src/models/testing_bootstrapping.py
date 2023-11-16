import torch
import torchvision
from .inception_model import InceptionV3
import timm
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score
sys.path.append('/home/21576262@su/masters/')
from src.data.get_data import get_her2test_dataset

def main():
    ##### SET PARAMETERS #####
    num_classes = 2
    batch_size = 32
    
    cd, custom_grn = define_colours()

    PATCH_SIZE=256
    STRIDE=PATCH_SIZE
    SEED=42
    num_cpus=8
    
    # model_names = {'occult-newt-137': 'RESNET34', 'fresh-firefly-138': 'RESNET18', 'morning-glitter-146': 'RESNET50', 'glamorous-firefly-147': 'INCEPTIONv3', 'magic-frost-148': 'INCEPTIONv4', '??' : 'INCEPTIONRESNETv2'}

    # Test each model
    for name in model_names.keys():
        scores = []
        slide_scores = []
        
        print(f"Testing model: {name}")
        # Load model
        model_path = os.path.join('masters/models/', name + '_model_weights.pth')
        model = load_trained_models(num_classes, model_path, model_names[name])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device) # send model to GPU
        
        InceptionResnet = True if model_names[model_name]=='INCEPTIONRESNETv2' else False
        Inception = True if (model_names[model_name]=='INCEPTIONv3' or model_names[model_name]=='INCEPTIONv4') else False
        
        # Get info on model data split
        json_path = os.path.join('masters/models/data_splits/', name + '.json')
        with open(json_path, 'r') as file:
            data = json.load(file)
        testing_folders = data['test']
        original_dataset = get_her2test_dataset(testing_folders, batch_size, Inception=isInception, InceptionResnet=isInceptionResnet, Resnet=isResnet) # original data, testing subset for this model
        n_samples=len(original_dataset)
        # bootstrapping 1000 iterations
        for i in range(500):
            # Get bootstrapped (sub)set of data
            bootstrapped_dataset = create_bootstrapped_dataset(dataset, n_samples)
            bootstrapped_dataloader = DataLoader(bootstrapped_dataset, batch_size=batch_size, num_workers=num_cpus, shuffle=False, drop_last=False)

            # Test model
            case_ids, true_labels, probabilities, predictions = test_model(model, bootstrapped_dataloader, device) 
            # patch-level AUC for samples in bootstrapped set
            auc_score = roc_auc_score(true_labels, model_predictions)
            scores.append(auc_score)
            
            # slide-level AUC
            
        
        if len(scores) != 1000:
            raise ValueError("Not enough AUC scores (<1000)")
        # all 1000 AUC scores for this model (for later inference)
        auc_scores = np.array(scores)
        file_name = os.path.join('/home/21576262@su/masters/reports/bootstrapping', 'auc_scores_' + name + '.csv')
        np.savetxt(file_name, auc_scores, delimiter=",")

def create_bootstrapped_dataset(dataset, n_samples):
    """
    Create a bootstrapped dataset by sampling with replacement.
    Parameters:
        dataset (Dataset): The original dataset.
        n_samples (int): The number of samples in the bootstrapped dataset.
    Returns:
        Dataset: A new dataset instance with bootstrapped data.
    """
    # Generate random indices with replacement
    indices = torch.randint(0, len(dataset), size=(n_samples,))
    # Subset the dataset with the generated indices
    bootstrapped_data = torch.utils.data.Subset(dataset, indices)
    return bootstrapped_data

def create_bootstrapped_dataset_for_slides(testing_folders, batch_size, Inception=isInception, InceptionResnet=isInceptionResnet, Resnet=isResnet):
    """
    Create a bootstrapped dataset by sampling with replacement.
    Parameters:
        dataset (Dataset): The original dataset.
        n_samples (int): The number of samples in the bootstrapped dataset.
    Returns:
        Dataset: A new dataset instance with bootstrapped data.
    """
    # Generate random indices with replacement
    indices = torch.randint(0, len(dataset), size=(n_samples,))
    # Subset from list of testing folders
    bootstrapped_folders = testing_folders[indices]
    
    
    return bootstrapped_data

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

def define_colours():
    M_darkpurple = '#783CBB'
    M_lightpurple = '#A385DB'
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

if __name__ == '__main__':
    main()