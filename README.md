# CNN Models for Multiclass Classification Problem with Subset of the CIFAR-10 Dataset

In this project, we'll be analyzing a subset of the **CIFAR-10 dataset** to develop a convolutional neural network (CNN) with PyTorch, aiming to classify images into one of three object categories: **airplane**, **automobile**, or **ship**.

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

Each image is a small, low-resolution RGB photo, making this a good dataset for testing lightweight models.

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

For further referencing, the paths for the training and test sets are:

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

![Base Model Architecture](images/base_model.png)

### Training
As done in class, `n_feature` is set to 5 and `p` is set o 0.3. 30 epochs were used for the training, and the training and validation curves are shown below:

![Base Model Training and Validation Curves](images/base_model_training.png)

### Evaluation

At the end of the training, an accuracy of 0.7877 was obtained, precision was 0.7916, and recall was 0.7877. The model shows good performance in terms of both precision and recall, with the values being relatively close, indicating that the model is effectively balancing the ability to correctly identify positive instances (recall) with the ability to avoid incorrectly labeling negatives as positives (precision). Accuracy is also decent, showing that the model correctly predicts the majority of instances.

The following confusion matrix was obtained:

![Base Model Confusion Matrix](images/base_model_confusion_matrix.png)

The model performance is acceptable, but with room for improvement. In the next section, we will change some model parameters and the model architecture with the goal to improve the model performance.

### Visualizing Filters

To better investigate the model behavior, we can visualize the filters of the convolutional layers of the model:

![Base Model First Convolution Layer](images/base_model_filters_conv1.png)

![Base Model Second Convolution Layer](images/base_model_filters_conv2.png)

### Visualizing Feature Maps

Also, we can visualize the outputs of the intermediate layers using hooks, generating the following feature map:

![Base Model Feature Maps](images/base_model_featuremap.png)

This feature map shows the output of the first convolutional layer (`conv1`) applied to 10 input images using 5 different filters (`fil#0` to `fil#4`).

- **Low-level feature extraction**: Filters capture basic visual patterns such as edges, textures, and shapes.
- **Filter specialization**: 
  - `fil#2` highlights object contours (e.g., aircraft edges).
  - `fil#1` responds more to bright regions (e.g., sky or background).
- **Preserved object structure**: Aircraft shapes remain recognizable, especially in high-contrast images.
- **Activation strength varies**: Brighter areas = stronger activation; darker = weak or no response.
- **Some filters less responsive**: `fil#4` produces weaker activations, possibly detecting subtle or redundant features.

--- 

## Other models

With the goal of improving the model performance, we developed 5 variations of the base model and performed the training + evaluation pipeline that was previously presented. The other models are as follows:

- Model 1: Base Model + `n_feature = 10`
- Model 2: Base Model + `n_feature = 15`
- Model 3: Base Model + `n_feature = 20`
- Model 4: Base Model + 1 Extra Convolution+Pooling Block (`n_feature = 5`)
- Model 5: Base Model + 2 Extra Convolution+Pooling Block (`n_feature = 5`)

As presented for the base model, the confusion matrix, accuracy, precision, recall, filter and feature maps visualization were made for all the models. For the sake of organization, we decide to not present all these results in this README. However, they all can be seen and be reproduced with the `part1.ipynb` notebook. We summarized the results in the following table:

| Model         | Trainable Parameters  | Accuracy  | Precision  | Recall    |
|---------------|-----------------------|-----------|------------|-----------|
| Base          | 6823                  | 0.7877    | 0.7916     | 0.7877    | 
| 1             | 13893                 | 0.8553    | 0.8599     | 0.8553    | 
| 2             | 21413                 | 0.8813    | 0.8814     | 0.8813    | 
| 3             | 29383                 | 0.8843    | 0.8847     | 0.8843    | 
| 4             | 8356                  | 0.6993    | 0.7192     | 0.6993    | 
| 5             | 10676                 | 0.8000    | 0.8004     | 0.8000    | 

For the first 3 models, for which the `n_feature` parameters varies, model 2 presents the best trade-off between trainable parameters and performance metrics, since model 3 presents a considerable increase in parameters for an improvement of only 0.0030 in accuracy. Between models 4 a 5, for which the architecture changes in relation to the base, model 5 presents a considerable increase in accuracy when adding two Convolution+Pooling blocks, instead of just one, which justifies the increase in trainable parameters.

## Final Model

For the final model, we decided to combine models 2 and 5, that is, a model with `n_feature=15` and 2 additional Convolution+Pooling blocks. The training and evaluation of the final model is presented in the next subsections.

### Training
The training and validation curves are shown below:

![Final Model Training and Validation Curves](images/final_model_training.png)

### Evaluation

At the end of the training, an accuracy of 0.8950 was obtained, precision was 0.8948, and recall was 0.0.8950. Accuracy showed a significant improvement when compared to the performance of the base model.

The following confusion matrix was obtained:

![Final Model Confusion Matrix](images/final_model_confusion_matrix.png)

### Visualizing Filters

To better investigate the model behavior, we can visualize the filters of the convolutional layers of the model:

![Final Model First Convolution Layer](images/final_model_filters_conv1.png)

![Final Model Second Convolution Layer](images/final_model_filters_conv2.png)

![Final Model First Convolution Layer](images/final_model_filters_conv3.png)

![Final Model Second Convolution Layer](images/final_model_filters_conv4.png)

### Visualizing Feature Maps

Also, we can visualize the outputs of the intermediate layers using hooks, generating the following feature map:

![Final Model Feature Maps](images/final_model_featuremap.png)

These feature maps illustrate the transition of the convolutional network from learning basic visual features to capturing more complex and abstract representations. In the `conv1` layer, the filters capture low-level features such as edges, contours, and simple textures. The activations are well-defined, and object shapes—such as aircraft—are still visible, indicating that the filters are focused on identifying fundamental structural patterns.

In contrast, the feature map that comes from a deeper layer in the network displays a much larger number of filters with denser, less sharp activations that are harder to interpret visually. This behavior is expected in deeper layers, where filters learn to combine features extracted earlier to form higher-level semantic representations, such as specific parts of objects or discriminative patterns important for classification tasks.

In summary, while the early layers of the network act as general detectors of basic visual patterns, the deeper layers serve as specialists, extracting increasingly abstract and task-relevant information. This hierarchical learning process is one of the key strengths of deep convolutional networks.

## **Conclusions**

In this project, we successfully applied a convolutional neural network (CNN) to classify a subset of the **CIFAR-10** dataset into three categories: **airplane**, **automobile**, and **ship**. The process included data preprocessing, model development, and evaluation. 

Through a series of model variations, we identified that the base CNN model performed well, achieving a solid balance between **precision** and **recall**. The **accuracy** of the base model was 0.7877, while the precision and recall were both around 0.7916 and 0.7877, respectively. However, after experimenting with various architectures, we achieved significant improvements. The **final model**, which combines the best aspects of both feature adjustments and architectural changes, yielded an accuracy of 0.8950, with corresponding precision and recall of 0.8948 and 0.8950. 

These results highlight the importance of exploring different configurations and hyperparameters to achieve optimal model performance. Notably, the model that added two extra convolutional blocks showed substantial improvements in accuracy and robustness.

While the model performs well on this simplified subset of the **CIFAR-10 dataset**, there is always room for further optimization, such as exploring more advanced architectures, hyperparameter tuning, or testing on larger datasets. This project provides a solid foundation for understanding and applying CNNs in the context of image classification.

### **Key Takeaways**:
- The **Base Model** offered a solid starting point, with good precision and recall values.
- Adjusting the **n_feature** parameter and adding **extra convolutional blocks** improved the model’s performance significantly.
- The **final model** (with `n_feature = 15` and two extra convolution blocks) provided the best trade-off between performance and computational complexity.

This work offers valuable insights into building and improving image classification models with PyTorch and can serve as a reference for further developments in machine learning applications.


## Intended Use

- **Educational**: Demonstrate multiclass classification model with CNN using PyTorch.
- **Not intended**: For production or real financial/customer decision-making systems.

---

## Ethical Considerations

- Dataset is synthetic: No real user data involved.

---

## How to Cite

```bibtex
@misc{customerpurchase2025,
  author = {Farias, Leandro and Lopes, Nicholas},
  title = {CNN Models for Multiclass Classification Problem with Subset of the CIFAR-10 Dataset},
  year = {2025},
  note = {Graduate project, UFRN - PPGEEC2318}
}
```
