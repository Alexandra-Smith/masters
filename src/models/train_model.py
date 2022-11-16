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
class AlexNet(nn.Module):
  # Define layers
  def __init__(self, num_classes):
    super(AlexNet, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
      nn.BatchNorm2d(96),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=2)
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(96, 256 ,kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )
    self.layer3 = nn.Sequential(
      nn.Conv2d(),
      nn.BatchNorm2d(384),
      nn.ReLU()
    )
    self.layer4 = nn.Sequential(
      nn.Conv2d(),
      nn.BatchNorm2d(),
      nn.ReLU()
    )
    self.layer5 = nn.Sequential(
      nn.Conv2d(),
      nn.BatchNorm2d(),
      nn.ReLU(),
      nn.MaxPool2d()
    )
    self.fc = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(9216, 4096),
      nn.ReLU()
    )
    self.fc1 = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU()
    )
    self.fc2 = nn.Sequential(
      nn.Linear(4096, num_classes)
    )

  # Define forward method
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out



# wandb.init(project="masters")

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

