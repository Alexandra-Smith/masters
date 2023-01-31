import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

def train_model(patches, labels, model, loss_func, optimiser, num_epochs):
    for epoch in range(num_epochs):
        running_loss=0.0
        for patch, label in zip(patches, labels):
            # Forward pass
            output = model(patch.unsqueeze(0))
            loss = loss_func(output, label.unsqueeze(0))

            # Zero gradients and perform backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Print running loss
            running_loss += loss.item()
        
        # Log average loss for epoch number to W&B
        wandb.log({'Loss': running_loss / len(patches), 'Epoch': epoch+1})