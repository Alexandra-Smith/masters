import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from .inception_model import InceptionV3

# Simple 6 layer CNN (1st implemented)
# DOI: 10.1038/srep46450
def CNN(num_classes):
    # Model parameters
    learning_rate = 0.001
    learning_rate_decay = 0.0000001
    parameters = {'learning_rate': learning_rate, 'learning_rate_decay': learning_rate_decay}

    # Define the model architecture
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 32 * 32, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )

    optimiser = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, None

# Pretrained Inception (2nd model)
# DOI:10.1038/s41598-018-27707-4
# (Wang)
def inception0(num_classes):
    # Model parameters
    learning_rate = 0.0001
    momentum = 0.9
    parameters = {'learning_rate': learning_rate, 'momentum': momentum}

    # Define the model architecture  
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True) # pre-trained model
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace last layers with new layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(2048, num_classes),
        nn.Softmax(dim=1)
    )
    
    # Set requires_grad=True for last 5 layers
    num_layers_to_train = 5
    ct = 0
    for name, child in model.named_children():
        if ct < num_layers_to_train:
            for param in child.parameters():
                param.requires_grad = True
        ct += 1
    
    # Set up optimiser
    optimiser = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum) # selectively update only the parameters that are being fine-tuned
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, None

# Pretrained ResNet
# https://doi.org/10.1038/s41598-019-40041-7
# https://github.com/BMIRDS/deepslide/blob/master/code/utils_model.py
# from Kather et al
def resnet18(num_classes):
    # Model parameters
    learning_rate = 1e-6
    weight_decay = 1e-4
    # learning_rate_decay = 0.85
    parameters = {"learning_rate": learning_rate, "weight_decay": weight_decay}

    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set the last 10 layers to trainable
    for param in list(model.parameters())[-10:]:
        param.requires_grad = True
        
#     # must be one of [18, 34, 50, 101, 152]
#     num_layers=18
#     model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
#     model = model_constructor(num_classes=num_classes)
#     # Modify to take any input
#     model.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Replace nn.AvgPool2d with nn.AdaptiveAvgPool2d
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
    
#     pretrained = model_constructor(pretrained=True).state_dict()
#     if num_classes != pretrained["fc.weight"].size(0):
#         del pretrained["fc.weight"], pretrained["fc.bias"]
#     model.load_state_dict(state_dict=pretrained, strict=False)

    optimiser = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay
                          )
    # scheduler = lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=learning_rate_decay)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    return model, optimiser, criterion, parameters, None

def INCEPTIONv3(num_classes):

    # Hyperparameters
    WEIGHT_DECAY = 0.9                  # Decay term for RMSProp.
    # weight_decay = 0.00004?
    # from inception_v3_parameters
    MOMENTUM = 0.9                      # Momentum in RMSProp.
    ALPHA = 1.0                       # Epsilon term for RMSProp.
    INITIAL_LEARNING_RATE = 0.1         # Initial learning rate.
    
#     NUM_EPOCHS_PER_DECAY = 30.0         # Epochs after which learning rate decays.
#     LEARNING_RATE_DECAY_FACTOR = 0.16   # Learning rate decay factor.

    # model = InceptionV3()
    model = InceptionV3(num_classes=num_classes)
    
    optimiser = optim.RMSprop(model.parameters(), 
                              lr=INITIAL_LEARNING_RATE, 
                              momentum=MOMENTUM, 
                              alpha=ALPHA, 
                              weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    parameters = {"learning_rate": INITIAL_LEARNING_RATE, "momentum": MOMENTUM, "alpha": ALPHA, 'RMS_weight_decay': WEIGHT_DECAY}

    return model, optimiser, criterion, parameters, None

def shufflenet():

    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)

    INITIAL_LEARNING_RATE=5e-5
    WEIGHT_DECAY=0.9
    EPSILON=1e-8

    optimiser = optim.Adam(params=model.parameters(),
                           lr=INITIAL_LEARNING_RATE,
                           eps=EPSILON,
                           weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    parameters = {"learning_rate": INITIAL_LEARNING_RATE, "epsilon": EPSILON, 'Adam_weight_decay': WEIGHT_DECAY}

    return model, optimiser, criterion, parameters, None

# Coudray 
# pretrained
def inceptionv3_preT(num_classes):
    
    # Hyperparameters
    WEIGHT_DECAY = 0.9                  # Decay term for RMSProp.
    MOMENTUM = 0.9                      # Momentum in RMSProp.
    ALPHA = 1.0                         # Epsilon term for RMSProp. (Alpha?)
    INITIAL_LEARNING_RATE = 0.1         # Initial learning rate.
    
    parameters = {"learning_rate": INITIAL_LEARNING_RATE, "momentum": MOMENTUM, "alpha": ALPHA, 'RMS_weight_decay': WEIGHT_DECAY}

    # Define the model architecture  
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True) # pre-trained model
    
    # Replace last layers with new layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(2048, num_classes),
    )
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Set requires_grad=True for last n layers
    num_layers_to_train = 3
    ct = 0
    for name, child in model.named_children():
        if ct < num_layers_to_train:
            for param in child.parameters():
                param.requires_grad = True
        ct += 1
    
    optimiser = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=INITIAL_LEARNING_RATE, 
                              momentum=MOMENTUM, 
                              alpha=ALPHA, 
                              weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, None