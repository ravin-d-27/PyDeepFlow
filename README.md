# **PyDeepFlow**

<p align="center">
  <img src="https://github.com/user-attachments/assets/81f3e52a-ad5a-47b5-a7e1-bdc9ee2de508" alt="logo" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/ravin-d-27/PyDeepFlow/stargazers">
    <img src="https://img.shields.io/github/stars/ravin-d-27/PyDeepFlow?style=social" alt="GitHub stars"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Open%20Source%20with%20Attribution-blue.svg" alt="License"/>
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.6%2B-blue.svg" alt="Python"/>
  </a>
</p>

---

## **What is PyDeepFlow?**

`pydeepflow` is a Python library designed for building and training deep learning models with an emphasis on **ease of use** and **flexibility**.  
It abstracts many of the complexities found in traditional deep learning libraries while still offering **powerful functionality**.

---

## **Hacktoberfest 2025 with PyDeepFlow 💙**

<p align="center">
  <img src="assets/HF2025-EmailHeader.png" alt="Hacktoberfest" width="80%"/>
</p>

<p align="center">
  Support open source software by participating in  
  <a href="https://hacktoberfest.com"><b>Hacktoberfest</b></a> 🎉  
  and get goodies and digital badges! 💙
</p>

---

## **Contributors**

Thanks to these amazing people for contributing to this project:

<p align="center">
  <a href="https://github.com/ravin-d-27/PyDeepFlow/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=ravin-d-27/PyDeepFlow" alt="Contributors"/>
  </a>
</p>



### **Key Features of Pydeepflow:**

- **Simplicity**: Designed for ease of use, making it accessible to beginners.
- **Configurability**: Users can easily modify network architectures, loss functions, and optimizers.
- **Flexibility**: Can seamlessly switch between CPU and GPU for training.

## **Why is Pydeepflow Better than TensorFlow and PyTorch?**

While TensorFlow and PyTorch are widely used and powerful frameworks, `pydeepflow` offers specific advantages for certain use cases:

1. **User-Friendly API**: `pydeepflow` is designed to be intuitive, allowing users to create and train neural networks without delving into complex configurations.
  
2. **Rapid Prototyping**: It enables quick testing of ideas with minimal boilerplate code, which is particularly beneficial for educational purposes and research.

3. **Lightweight**: The library has a smaller footprint compared to TensorFlow and PyTorch, making it faster to install and easier to use in lightweight environments.

4. **Focused Learning**: It provides a straightforward approach to understanding deep learning concepts without getting bogged down by the extensive features available in larger libraries.

## **Dependencies**

The project requires the following Python libraries:

- `numpy`: For numerical operations and handling arrays.
- `pandas`: For data manipulation and loading datasets.
- `scikit-learn`: For splitting data and preprocessing.
- `tqdm`: For progress bars in training.
- `jupyter`: (Optional) For working with Jupyter notebooks.
- `pydeepflow`: The core library used to implement the Multi-Layer ANN.

You can find the full list in `requirements.txt`.

## **How to Install and Use Pydeepflow from PyPI**

### **Installation**

You can install `pydeepflow` directly from PyPI using pip. Open your command line and run:

```bash
pip install pydeepflow
```

### **Using Pydeepflow**

After installing, you can start using `pydeepflow` to create and train neural networks. Below is a brief example:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pydeepflow.model import Multi_Layer_ANN
from pydeeepflow.datasets import load_iris

# Load Iris dataset
df = load_iris(as_frame=True)

# Data preprocessing
df['species'] = df['species'].astype('category').cat.codes
X = df.iloc[:, :-1].values
y = np.eye(len(np.unique(y)))[y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train ANN
ann = Multi_Layer_ANN(X_train, y_train, hidden_layers=[5, 5], activations=['relu', 'relu'], loss='categorical_crossentropy')
ann.fit(epochs=1000, learning_rate=0.01)

# Evaluate
y_pred = ann.predict(X_test)
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

PyDeepFlow now also supports regression tasks. Here is how you can train a model and evaluate it with the new regression metrics:

# Create a simple regression dataset
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = (0.5 * X**2 + 2 * X + 5 + np.random.randn(100, 1) * 5).reshape(-1, 1)

# Split and scale data (similar to classification example)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)
scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

# Train ANN for regression
ann_regression = Multi_Layer_ANN(X_train, y_train, hidden_layers=[10, 10], activations=['relu', 'relu'], loss='mean_squared_error')
ann_regression.fit(epochs=500, learning_rate=0.01)

# Evaluate the model using new regression metrics
print("Regression Model Evaluation:")
regression_results = ann_regression.evaluate(
    X_test,
    y_test,
    metrics=['mean_absolute_error', 'mean_squared_error', 'r2_score']
)
print(regression_results)


## **Transfer Learning with VGG16** 🚀

PyDeepFlow now supports **transfer learning** with the VGG16 architecture! This powerful feature allows you to leverage pretrained deep learning models for your custom computer vision tasks.

### **What is VGG16?**

VGG16 is a deep convolutional neural network with 16 layers (13 convolutional + 3 fully connected) that was developed by the Visual Geometry Group at Oxford. It's widely used for image classification and feature extraction tasks.

### **Quick Start with VGG16**

```python
from pydeepflow.pretrained import VGG16
from pydeepflow.transfer_learning import TransferLearningManager
import numpy as np

# Load VGG16 for your custom dataset (e.g., 10 classes)
vgg = VGG16(num_classes=10, freeze_features=True)

# Display architecture
vgg.summary()

# Use Transfer Learning Manager for structured workflow
manager = TransferLearningManager(vgg)

# Phase 1: Feature Extraction (train only classifier)
manager.setup_feature_extraction()
# Train your model here with frozen conv layers...

# Phase 2: Fine-Tuning (unfreeze last conv block)
manager.setup_fine_tuning(num_layers=3)
# Continue training with unfrozen layers...
```

### **Transfer Learning Workflows**

#### **1. Feature Extraction (Small Datasets)**
Best for datasets with < 1000 samples:

```python
# Freeze all convolutional layers
vgg = VGG16(num_classes=5, freeze_features=True)

# Train only the classifier
# Recommended: 10-20 epochs, LR = 1e-2
```

#### **2. Fine-Tuning (Medium to Large Datasets)**
Best for datasets with > 1000 samples:

```python
# Start with frozen features
vgg = VGG16(num_classes=10, freeze_features=True)
# Train classifier first...

# Then unfreeze last conv block for fine-tuning
vgg.unfreeze_layers(num_layers=3)
# Continue training with lower LR (1e-3 to 1e-4)
```

#### **3. Feature Extraction Only**
Use VGG16 as a feature extractor for other classifiers:

```python
# Create VGG16 without classifier layers
vgg_features = VGG16(include_top=False)

# Extract features
features = vgg_features.predict(X_images)

# Use features with SVM, Random Forest, or custom classifier
```

### **Key Features**

- ✅ **Full VGG16 Architecture**: 13 conv layers + 3 FC layers
- ✅ **Layer Freezing/Unfreezing**: Fine-grained control over trainable layers
- ✅ **Transfer Learning Manager**: Structured workflow for best practices
- ✅ **Weight Save/Load**: Save and load pretrained weights
- ✅ **GPU Support**: Accelerate training with CUDA
- ✅ **Progressive Unfreezing**: Gradually unfreeze layers to prevent catastrophic forgetting

### **Advanced Usage**

```python
from pydeepflow.transfer_learning import (
    calculate_trainable_params,
    print_transfer_learning_guide
)

# Get detailed parameter information
params = calculate_trainable_params(vgg)
print(f"Trainable: {params['trainable']:,} / {params['total']:,}")

# Display best practices guide
print_transfer_learning_guide()

# Progressive unfreezing strategy
manager = TransferLearningManager(vgg)
stages = manager.progressive_unfreeze(stages=3)

for stage_info in stages:
    vgg.unfreeze_layers(num_layers=stage_info['layers_to_unfreeze'])
    # Train with recommended learning rate...
```

### **Examples**

Check out the `examples/vgg16_transfer_learning.py` file for comprehensive examples including:
- Basic VGG16 usage
- Feature extraction workflow
- Fine-tuning strategies
- Progressive unfreezing
- Feature extraction without classifier

### **Tests**

Run VGG16 tests to verify functionality:

```bash
python -m pytest tests/test_vgg16.py -v
```

## **GPU and CPU Support**

`PyDeepFlow` is designed to be flexible and can run on both CPUs and NVIDIA GPUs.

### **CPU-Only (Default)**

By default, `PyDeepFlow` uses `NumPy` for all computations and will run on your CPU. The standard installation is all you need:

```bash
pip install pydeepflow
```

### **GPU Acceleration**
If you have an NVIDIA GPU and have installed the CUDA toolkit, you can enable GPU acceleration by installing `CuPy`.
To install the library with GPU support, use the following command:

```bash
pip install pydeepflow[gpu]
```

If you try to use the GPU functionality without `CuPy` installed, the library will print a warning and safely fall back to using the CPU.



## **Contributing to Pydeepflow on GitHub**

Contributions are welcome! If you would like to contribute to `pydeepflow`, follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page.
  
2. **Clone Your Fork**: Use git to clone your forked repository:
   ```bash
   git clone https://github.com/ravin-d-27/PyDeepFlow.git
   cd pydeepflow
   ```

3. **Create a Branch**: Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b my-feature-branch
   ```

4. **Make Your Changes**: Implement your changes and commit them:
   ```bash
   git commit -m "Add some feature"
   ```

5. **Push to Your Fork**:
   ```bash
   git push origin my-feature-branch
   ```

6. **Submit a Pull Request**: Go to the original repository and submit a pull request.

## **References**

- **Iris Dataset**: The dataset used in this project can be found at the UCI Machine Learning Repository: [Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/)
  
- **pydeepflow Documentation**: [pydeepflow Documentation](https://pypi.org/project/pydeepflow/)

- **Deep Learning Resources**: For more about deep learning, consider the following:
  - Goodfellow, Ian, et al. *Deep Learning*. MIT Press, 2016.
  - Chollet, François. *Deep Learning with Python*. Manning Publications, 2017.

## **Author**

**Author Name**: Ravin D  
**GitHub**: [ravin-d-27](https://github.com/ravin-d-27)  
**Email**: ravin.d3107@outlook.com
<br><br>
The author is passionate about deep learning and is dedicated to creating tools that make neural networks more accessible to everyone.

---
