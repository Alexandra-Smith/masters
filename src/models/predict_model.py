import numpy as np
import cv2
import torch

# Load in trained model
model = 

# Get test data
X_test = 
y_test = 

# Predict a label for each patch using the trained model
predictions = []
for patch in X_test:
    patch = torch.from_numpy(patch).unsqueeze(0)
    output = model(patch)
    prediction = torch.argmax(output).item()
    predictions.append(prediction)

# Convert the predictions to a 2D array to match the shape of the input image
height, width = ... # Shape of the input image
predictions = np.array(predictions).reshape(height, width)

# Create a heat map image with the final predictions
heatmap = cv2.applyColorMap(predictions.astype(np.uint8), cv2.COLORMAP_JET)