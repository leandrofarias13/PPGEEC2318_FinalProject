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

'''python
path_train = 'cifar10_subset/train'
print("Path to dataset files for training:", path_train)

path_test = 'cifar10_subset/test'
print("Path to dataset files for test:", path_test)
```
