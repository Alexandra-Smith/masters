from initialise_models import *
from train_model import *
# from predict_model import *
# import torch.utils.data as data
# import torch.nn.functional as F
from torchvision import transforms
import torch.utils.data as data_utils
import sys
# sys.path.append('/Users/alexandrasmith/Desktop/Workspace/Projects/masters/src/')
# from data.save_data import PATCH_SIZE, STRIDE, NUM_CLASSES, SAVE_DEST

SAVE_DEST = '/Users/alexandrasmith/Desktop/Workspace/Projects/masters/data/processed/tests/'
PATCH_SIZE=1024
STRIDE=PATCH_SIZE
NUM_CLASSES=2

# Initialise new run on W&B
# wandb.init(project="masters")

# Load saved data
patches = torch.load(SAVE_DEST + "00-patches.pt")
labels = torch.load(SAVE_DEST + "00-labels.pt")

# Images have to be in range [0, 1] (saved tensors already are)
# And need to have size at least 299x299
# Normalise all input images to specific range
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_input = preprocess(patches)

# trainloader = DataLoader(model_input, batch_size=32, shuffle=True, drop_last=True)
train = data_utils.TensorDataset(patches, labels)
train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True, drop_last=True)

# Define model variable
NUM_EPOCHS = 20
LEARNING_RATE = 0.1
EPSILON=1.0
WEIGHT_DECAY=0.9
MOMENTUM = 0.9

# Define model
inception_v3_model, loss_func, optimiser = initialise_inceptionv3_model(NUM_CLASSES, LEARNING_RATE, EPSILON, WEIGHT_DECAY, MOMENTUM)
# print(inception_v3_model)

# Training
train_model(train_loader, inception_v3_model, loss_func, optimiser, NUM_EPOCHS)

# Finalise the W&B run
# wandb.join()

# save the trained model
# torch.save(inception_v3_model.state_dict(), "00-trained_model.pt")

# Make predictions