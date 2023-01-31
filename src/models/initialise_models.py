import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

def initialise_inceptionv3_model(num_classes: int, learning_rate, momentum):
    '''
    Function to initialise the inceptionv3 model
    '''
    # Define model
    model = models.inception_v3(pretrained=True)

    # Replace the final layer of the model with a single output node
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze the parameters of the model to prevent backpropagation through them (all layers except for final fully connected layer)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True

    # Define the loss function and optimizer
    loss_func = nn.BCELoss()
    optimiser = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=momentum)

    return model, loss_func, optimiser