import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import time
import copy

# Load the dataset

# Define variables
NUM_CLASSES=3 #background, normal, malignant
BATCH_SIZE=32
NUM_EPOCHS=20
FEATURE_EXTRACT=True

# Define helper functions
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    '''
    Handles training and validation of model
    --------------
    Takes in a PyTorch model, a dictionary of dataloaders, a loss function, an optimiser,
    a specific number of epochs to train and validate for, boolean flag for when model is an Inception model
    '''
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    '''
    Sets .requires_grad attribute to False when feature extracting
    By default when we load a pretrained model all parameters have .requires_grad=True
    (fine for training from scratch/finetuning)
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    """ Initialise Inception v3
    Be careful, expects (299,299) sized images and has auxiliary output
    """
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299

    return model_ft, input_size


# Set device to run training on GPU or CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model for this run
model_ft, input_size = initialize_model(NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)
print("Done")

# inception_v3 = models.inception_v3(pretrained=True)

# Initialise new run in wandb database
# wandb.init(name="run_name", project="masters", notes="Test run 1", entity='')

# # Capture a dictionary of hyperparameters with config
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }

# # Log metrics inside your training loop to visualize model performance
# wandb.log({"loss": loss})

# # Optional
# wandb.watch(model)

