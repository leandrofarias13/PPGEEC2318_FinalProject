# Multiclass Classification Problem with Subset of the CIFAR-10 Dataset

In this notebook, we'll be analyzing a subset of the **CIFAR-10 dataset** to develop a convolutional neural network (CNN) with PyTorch, aiming to classify images into one of three object categories: **airplane**, **automobile**, or **ship**.

This work was conducted as part of the *Machine Learning* course (PPGEEC2318) of the Graduate Program in Electrical and Computer Engineering at UFRN. 

> **Professor:** Ivanovitch M. Silva  
> **Students:**  
> Leandro Roberto Silva Farias – 20251011748  
> Nicholas Medeiros Lopes – 20251011739

The goal is to train and evaluate different CNN models to perform the task described above. All models are derived from the model seen in class, which will referred to as **Base Model**. 

The complete pipeline includes fetching data, preprocessing, data preparation, model training, model evaluation, and reporting. The complete pipeline is contained in the `part1.ipynb` file.

---

## Environment Setup

The following libraries are required to run the code:

```python
# Import standard libraries for randomness, deep copying, and numerical operations
import random
import numpy as np
from copy import deepcopy

# Import libraries for image processing and data manipulation
from PIL import Image
import pandas as pd

# Import PyTorch core and utilities for deep learning
import torch
import torch.optim as optim  # Optimization algorithms
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for non-parametric operations

# Import PyTorch utilities for data loading and transformations
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision.transforms.v2 import Compose, ToImage, Normalize, ToPILImage, Resize, ToDtype

# Import dataset handling and learning rate schedulers
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CyclicLR, LambdaLR

# Import visualization and web utilities
import matplotlib.pyplot as plt
import requests
import zipfile
import os
import errno
import kagglehub
import seaborn as sns

# Import model evaluation utilities
from sklearn.metrics import confusion_matrix

# Set matplotlib style for better visuals
plt.style.use('fivethirtyeight')
```

---

## Dataset Description

The **CIFAR-10** dataset is a well-known benchmark in computer vision, consisting of 60,000 color images (32×32 pixels) across 10 different classes. However, for the purposes of this project, we selected only the following 3 classes:

- **Airplane**  
- **Automobile**  
- **Ship**  

Each image is a small, low-resolution RGB photo, making this a good dataset for testing lightweight models and data augmentation strategies.

The **target variable** is the class label of the image, encoded as:

- **0 = Airplane**  
- **1 = Automobile**  
- **2 = Ship**  

This setup characterizes a **multiclass classification problem with 3 classes**, rather than the full 10-class setup from the original dataset. The selected subset is **balanced**, with roughly equal numbers of samples per class.

We'll preprocess the data, perform visualization, and train deep learning models to distinguish between these transport-related categories using visual cues.

---

## Fetch Data

The full CIFAR-10 dataset is available at [CIFAR-10 - Kaggle Dataset](https://www.kaggle.com/datasets/ayush1220/cifar10). The simplified version of this dataset, with only the **airplaine**, **automobile** and **ship** classes, is in the file `data/cifar10_subset.zip` of this repository. After downloading the file and uploading it to your Google Colaboratory environment, the dataset can be extracted with the following commands:

```python
with zipfile.ZipFile("cifar10_subset.zip", 'r') as zip_ref:
        zip_ref.extractall('')
```

For further referencing, the paths for the training and test sets is recorded as follows:

```python
path_train = 'cifar10_subset/train'
print("Path to dataset files for training:", path_train)

path_test = 'cifar10_subset/test'
print("Path to dataset files for test:", path_test)
```

The first images of the train set for each class are shown below as examples:

![Dataset Examples](images/dataset_examples.png)

---

## Data Preprocessing

The preprocessing starts by defining a pipeline using `torchvision.transforms` that performs the following operations:

1. **Resize** all images to `28x28` pixels  
2. **Convert** to a `PIL.Image` object to ensure consistency in channels  
3. **Cast to float** and **normalize** pixel values from the `[0, 255]` range to `[0.0, 1.0]`  

These transformations are composed using `Compose()` and applied to the dataset through `ImageFolder`, which automatically labels images based on subfolder names.

```python
temp_transform = Compose([
    Resize(28),                      # Resize each image to 28x28
    ToImage(),                       # Convert tensor back to PIL Image (RGB)
    ToDType(torch.float32, scale=True)  # Convert to float32 and normalize pixel values to [0,1]
])
```

The dataset is then loaded:

```python
temp_dataset = ImageFolder(
    root=path_train,
    transform=temp_transform  # Apply the preprocessing pipeline to every image
)
```

Normalization improves training by ensuring that input values have zero mean and unit variance. To compute normalization values:

```python
temp_loader = DataLoader(temp_dataset, batch_size=16)
first_images, first_labels = next(iter(temp_loader))
Architecture.statistics_per_channel(first_images, first_labels)
```

We apply this across the entire dataset:

```python
results = Architecture.loader_apply(temp_loader, Architecture.statistics_per_channel)
```

This gives the sums of means and standard deviations, which can be used to compute averages:

```python
normalizer = Architecture.make_normalizer(temp_loader)
```

The dataset is split into training and validation sets using folder structure:

```python
composer = Compose([
    Resize(28),
    ToImage(),
    ToDType(torch.float32, scale=True),
    normalizer  # Apply normalization transform
])

train_data = ImageFolder(root=path_train, transform=composer)
val_data   = ImageFolder(root=path_test, transform=composer)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)
```

After the preprocessing, we define a function to visualize some sample images:

![Example of Preprocessed Images](images/preprocessing_example.png)

---

## Base CNN Model

### Architecture
The base model used in this project, which was presented in the PPGEEC2318 classes, employs two convolutional blocks, each consisting of a convolutional layer followed by a ReLU activation function and a max-pooling layer. Below are the key features of the architecture:

![Base Model Architecture](images/base_model.png)

1. **Convolutional Layers**:
   - **Conv1**: The first convolutional layer applies 3x3 filters to the input image, which has 3 channels (RGB), and outputs feature maps with `n_feature` channels. This layer is followed by the ReLU activation function, which introduces non-linearity into the model. After the activation, max-pooling with a 2x2 kernel is applied, reducing the spatial dimensions of the feature maps.
   - **Conv2**: The second convolutional layer takes the output from the first layer and applies 3x3 filters, again followed by ReLU and max-pooling with a 2x2 kernel. The number of feature maps is kept the same as the previous layer.

2. **Feature Extraction**:
   - After the two convolutional blocks, the output feature maps are flattened to a one-dimensional vector, which is used as the input to the fully connected layers.

3. **Fully Connected Layers**:
   - **fc1**: The first fully connected layer consists of 50 units and receives the flattened feature vector from the convolutional layers. This layer is followed by the ReLU activation function. Dropout is applied here to prevent overfitting, with a probability `p` (if greater than 0).
   - **fc2**: The second fully connected layer has 3 output units, corresponding to the number of classes in the classification task. This layer provides the final classification result.

4. **Dropout**:
   - Dropout layers are added after the fully connected layers with the specified dropout probability `p`. This helps prevent overfitting by randomly setting some of the weights to zero during training.

5. **Overall Structure**:
   - The network takes an image as input, applies two convolutional blocks to extract hierarchical features, and then uses fully connected layers to classify the image into one of the predefined classes. The final output is a vector of size 3, representing the predicted class probabilities for a 3-class classification problem.

### Training
As done in class, `n_feature` is set to 5 and `p` is set o 0.3.
