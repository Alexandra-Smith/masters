import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from .inception_model import InceptionV3
from torchvision import models
import timm
import math
from torch.optim.lr_scheduler import _LRScheduler

class ExponentialDecayStep(_LRScheduler):
    def __init__(self, optimiser, decay_factor, decay_step, last_epoch=-1):
        self.decay_factor = decay_factor
        self.decay_step = decay_step
        super(ExponentialDecayStep, self).__init__(optimiser, last_epoch)

    def get_lr(self):
        return [base_lr * (self.decay_factor ** (self.last_epoch // self.decay_step))
                for base_lr in self.base_lrs]

# Simple 6 layer CNN (1st implemented)
# DOI: 10.1038/srep46450
def CNN(num_classes, checkpoint_path=None):
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
    
    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt)

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

    # # Define the model architecture  
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', weights='DEFAULT') # pre-trained model
    
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
def resnet18(num_classes, checkpoint_path=None):
    # Model parameters
    learning_rate = 1e-6
    weight_decay = 1e-4
    # learning_rate_decay = 0.85
    # parameters = {"learning_rate": learning_rate, "weight_decay": weight_decay}
    parameters = {"learning_rate": learning_rate}

    model = models.resnet18(weights='DEFAULT')
    
     # Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc = nn.Sequential(
        nn.Dropout(p=0.7),
        nn.Linear(model.fc.in_features, num_classes),
        )
    
    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt)

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
                          )
    # scheduler = lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=learning_rate_decay)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()

    return model, optimiser, criterion, parameters, None

def resnet50(num_classes):
    # Model parameters
    learning_rate = 1e-6
    # weight_decay = 1e-4
    # learning_rate_decay = 0.85
    parameters = {"learning_rate": learning_rate}

    model = models.resnet50(weights='DEFAULT')
    
     # Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Set the last 10 layers to trainable
    for param in list(model.parameters())[-10:]:
        param.requires_grad = True
        
    # Setup optimiser
    optimiser = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                          )
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, None



# FULLY TRAINED
def resnet18full(num_classes):
    # Model parameters
    learning_rate = 1e-6
    weight_decay = 1e-4
    parameters = {"learning_rate": learning_rate, "weight_decay": weight_decay}

    model = torchvision.models.resnet18()
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    optimiser = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay
                          )
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    return model, optimiser, criterion, parameters, None


def INCEPTIONv3(num_classes, checkpoint_path=None):

    # Hyperparameters
    # WEIGHT_DECAY = 0.9                  # Decay term for RMSProp.
    # # weight_decay = 0.00004?
    # # from inception_v3_parameters
    # MOMENTUM = 0.9                      # Momentum in RMSProp.
    # EPSILON = 1.0                       # Epsilon term for RMSProp.
    # INITIAL_LEARNING_RATE = 0.1         # Initial learning rate.
    # NUM_EPOCHS_PER_DECAY = 30.0         # Epochs after which learning rate decays.
    # LEARNING_RATE_DECAY_FACTOR = 0.16   # Learning rate decay factor.

    # weight_decay=0.9
    # momentum=0.9
    # epsilon=1.0
    
    # learning_rate =  0.0001
    # initial_learning_rate=0.1
    # learning_rate_decay=0.9
    
    # Coudray
    initial_learning_rate=0.1
    learning_rate_decay=0.16
    
    # Gamble
    # initial_learning_rate=0.0055
    # learning_rate_decay=0.9
    
    # initial_learning_rate=0.0055
    # learning_rate_decay=0.16

    momentum=0.9
    epsilon=1
    rms_decay=0.9
    
    weight_decay=4e-05 # L2
    
    # learning_rate=0.0001
    # momentum=0.9
    # parameters = {"learning_rate": learning_rate,
    #               "momentum": momentum
    #              }
    parameters = {"learning_rate": initial_learning_rate,
                  "learning_rate_decay": learning_rate_decay,
                  "momentum": momentum, 
                  "epsilon": epsilon, 
                  "RMS_decay/alpha": rms_decay,
                  "weight_decay/L2": weight_decay}
    
    model = InceptionV3(num_classes=num_classes)
    
    if checkpoint_path != None:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt)
        print("checkpoint path loaded")
        
    # Freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set the last n layers to trainable
    for param in list(model.parameters())[-15:]:
        param.requires_grad = True
    
    optimiser = optim.RMSprop(model.parameters(), 
                              lr=initial_learning_rate,
                              alpha=rms_decay,
                              momentum=momentum, 
                              eps=epsilon, 
                              weight_decay=weight_decay)
    # optimiser = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=learning_rate_decay)
    
    criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, scheduler


def shufflenet():

    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', weights='DEFAULT')

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
def inceptionv3_pretrained(num_classes):
    
    lr=0.0001
    momentum=0.9
    parameters = {"learning_rate": lr,
                  "momentum": momentum
                 }
    
    # Define the model architecture  
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', weights='DEFAULT') # pre-trained model
    
     # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace last layers with new layers
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 2048),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.7),
    #     nn.Linear(2048, num_classes),
    # )
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2048),
        nn.Dropout(p=0.7),
        nn.Linear(2048, num_classes)
    )
    
    # Set the last n layers to trainable
    for param in list(model.parameters())[-1:]:
        param.requires_grad = True
    
    optimiser = optim.RMSprop(model.parameters(), 
                              lr=INITIAL_LEARNING_RATE, 
                              momentum=MOMENTUM, 
                              eps=EPSILON, 
                              weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Create a learning rate scheduler.
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=LEARNING_RATE_DECAY_FACTOR)

    return model, optimiser, criterion, parameters, scheduler

def inceptionresnetv2(num_classes, checkpoint_path=None):
    
    # learning_rate=1e-5
    
    learning_rate=0.0055
    learning_rate_decay=0.9
    
    weight_decay=4e-05
    
    parameters = {"learning_rate": learning_rate,
                  "learning_rate_decay": learning_rate_decay,
                  "weight_decay/L2": weight_decay}
    
    # parameters = {"learning_rate": learning_rate,
    #               "weight_decay": weight_decay}
                  
    model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=num_classes)
    
    # # Freeze all parameters
    # for param in model.parameters():
    #     param.requires_grad = False
    
    model.classif = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.7),
        nn.Linear(model.get_classifier().in_features, num_classes)
    )
    # # Set the last n layers to trainable
    # for param in list(model.parameters())[-10:]:
    #     param.requires_grad = True
    
    optimiser = optim.Adam(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=weight_decay)
    
    # optimiser = optim.RMSprop(model.parameters(), 
    #                           lr=initial_learning_rate,
    #                           alpha=rms_decay,
    #                           momentum=momentum, 
    #                           eps=epsilon, 
    #                           weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=learning_rate_decay)
                           
    criterion = nn.CrossEntropyLoss()
    
    
    ####################################################
#     initial_learning_rate=0.0055
#     learning_rate_decay=0.16
    
#     momentum=0.9
#     epsilon=1
#     rms_decay=0.9
    
#     weight_decay=4e-05 # L2
    
#     parameters = {"learning_rate": initial_learning_rate,
#                   "learning_rate_decay": learning_rate_decay,
#                   "momentum": momentum, 
#                   "epsilon": epsilon, 
#                   "RMS_decay/alpha": rms_decay,
#                   "weight_decay/L2": weight_decay}

# #     initial_learning_rate = 0.0002
# #     learning_rate_decay = 0.7
# #     weight_decay=4e-05 # L2
    
# #     parameters = {"learning_rate": initial_learning_rate,
# #                   "learning_rate_decay": learning_rate_decay,
# #                   "weight_decay/L2": weight_decay}
    
#     # model = timm.create_model('inception_v4', pretrained=True, num_classes=num_classes)    
#     model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)
#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad = False

#     # model.fc = nn.Sequential(
#     #     nn.Linear(model.fc.in_features, 1000),
#     #     nn.Dropout(p=0.7),
#     #     nn.Linear(1000, num_classes)
#     # )
#     # Set the last n layers to trainable
#     for param in list(model.parameters())[-3:]:
#         param.requires_grad = True
        
#     # num_ftrs = model.fc.in_features
#     # model.fc = nn.Linear(num_ftrs, num_classes)
    
#     # optimiser = optim.Adam(model.parameters(), 
#     #                        lr=initial_learning_rate, 
#     #                        weight_decay=weight_decay)

#     # Create the custom learning rate scheduler
#     # scheduler = ExponentialDecayStep(optimiser, decay_factor=learning_rate_decay, decay_step=2)
    
#     optimiser = optim.RMSprop(model.parameters(), 
#                               lr=initial_learning_rate,
#                               alpha=rms_decay,
#                               momentum=momentum, 
#                               eps=epsilon, 
#                               weight_decay=weight_decay)
#     # optimiser = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
#     scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=learning_rate_decay)
    
#     criterion = nn.CrossEntropyLoss()

    return model, optimiser, criterion, parameters, None