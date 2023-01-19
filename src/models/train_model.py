import wandb
import torch
import torch.nn as nn
import torchvision

# Load the dataset

# Define variables
batch_size=32
learning_rate=0.001
num_classes=3 #background, normal, malignant
num_epochs=20

# Set device to run training on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# wandb.init(name="run_name", project="masters")

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

