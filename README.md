# Skin Lesion Image Classification

## Project Overview

This project focuses on classifying skin lesion images into 8 categories using deep learning techniques. We implemented both a Convolutional Neural Network (CNN) from scratch and leveraged transfer learning with fine-tuning using ResNet-50 to improve classification accuracy.

## Dataset

We used a dataset consisting of 25,331 skin lesion images, categorized into the following classes:

- **Melanoma (MEL)**
- **Melanocytic nevus (NV)**
- **Basal cell carcinoma (BCC)**
- **Actinic keratosis (AK)**
- **Benign keratosis (BKL)**
- **Dermatofibroma (DF)**
- **Vascular lesion (VASC)**
- **Squamous cell carcinoma (SCC)**

## Methods

### 1. Convolutional Neural Network (CNN)

We designed and trained a CNN using the PyTorch framework. The architecture consists of:

- Two convolutional layers with ReLU activation and max-pooling
- Three fully connected layers
- Output layer with 8 classes

### Training Process

The training followed these steps:

- Zeroing gradients
- Forward pass
- Computing cross-entropy loss
- Backpropagation
- Optimizer step
- Computing predictions
- Updating the confusion matrix

**Results:**

- Training accuracy: 67.8%
- Testing accuracy: 65.8%

### 2. Transfer Learning with ResNet-50

We applied transfer learning using a pre-trained ResNet-50 model, modifying the last fully connected layer to output 8 classes. The training process involved hyperparameter tuning and optimization using the Adam optimizer with a OneCycleLR scheduler.

**Results:**

- Training accuracy: 83.22%
- Testing accuracy: 83.22%

## Code Implementation

### CNN Model (PyTorch)

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 53 * 53, 1200)
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, 8)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
# Transfer Learning with ResNet-50

## Code Implementation

```python
import torch
from torchvision import models

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the last fully connected layer to output 8 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)

# Move the model to the appropriate device (GPU/CPU)
model = model.to(device)
```

# Conclusion
- The CNN model achieved an accuracy of 65.8%.
- The ResNet-50 model significantly improved performance, reaching 83.22% accuracy.
- Transfer learning proved to be an effective method, leveraging pre-trained models to enhance classification performance.
# Authors
- Chaimae HADROUCH
- Kaoutar AIT AHMAD

# Date
March 6, 2022
