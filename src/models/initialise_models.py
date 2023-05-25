import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision

# Simple 6 layer CNN (1st implemented)
# DOI: 10.1038/srep46450
def NN(num_classes):
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

    return model, optimiser, criterion, parameters

# Pretrained Inception (2nd model)
# DOI:10.1038/s41598-018-27707-4
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

    return model, optimiser, criterion, parameters

# Pretrained ResNet
# https://doi.org/10.1038/s41598-019-40041-7
# https://github.com/BMIRDS/deepslide/blob/master/code/utils_model.py
def resnet(num_classes):
    # Model parameters
    learning_rate = 0.001
    weight_decay = 1e-4
    learning_rate_decay = 0.85
    parameters = {"learning_rate": learning_rate, "learning_rate_decay": learning_rate_decay, "weight_decay": weight_decay}

    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
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
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimiser,
                                           gamma=learning_rate_decay)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, scheduler