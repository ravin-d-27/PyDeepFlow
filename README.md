

---

# PyDeepFlow Documentation

## Overview

**PyDeepFlow** is a deep learning package optimized for performing deep learning tasks. It is designed to be easy to learn and integrate into various projects. The package provides a set of utilities for model building, training, and evaluation, making it a suitable choice for both beginners and experienced practitioners.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Model Training](#model-training)
  - [Evaluating the Model](#evaluating-the-model)
- [API Reference](#api-reference)
  - [Modules](#modules)
  - [Classes and Functions](#classes-and-functions)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Easy-to-use API for deep learning tasks.
- Support for various activation functions, loss functions, and optimizers.
- Compatibility with Jupyter notebooks for interactive usage.
- Flexible model training and evaluation utilities.
- Extensive documentation and examples for quick start.

## Installation

To install `PyDeepFlow`, you can use `pip`. Ensure you have Python 3.6 or higher installed. You can install the package from PyPI using the following command:

```bash
pip install pydeepflow
```

To install the package along with its dependencies, you can run:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

Here’s a quick example of how to use `PyDeepFlow` to create and train a simple model.

```python
from pydeepflow.model import Model
from pydeepflow.losses import MeanSquaredError
from pydeepflow.activations import ReLU
import numpy as np

# Generate some random data
X = np.random.rand(100, 10)
y = np.random.rand(100, 1)

# Create a model
model = Model()

# Add layers to the model
model.add_layer(units=64, activation=ReLU())
model.add_layer(units=1, activation=None)

# Compile the model
model.compile(loss=MeanSquaredError(), optimizer='adam')

# Train the model
model.fit(X, y, epochs=10, batch_size=16)
```

### Model Training

To train a model, you can call the `fit` method on your model instance. Here’s a more detailed explanation of the parameters:

- `X`: Input data (numpy array).
- `y`: Target data (numpy array).
- `epochs`: Number of times to iterate over the training data.
- `batch_size`: Number of samples per gradient update.

```python
model.fit(X, y, epochs=20, batch_size=32)
```

### Evaluating the Model

After training the model, you can evaluate its performance using the `evaluate` method.

```python
loss = model.evaluate(X, y)
print(f'Model Loss: {loss}')
```

## API Reference

### Modules

- **activations**: Contains various activation functions.
- **losses**: Contains loss functions to measure the model performance.
- **model**: Defines the model structure and training procedures.
- **device**: Provides functionalities related to device management (e.g., GPU support).

### Classes and Functions

#### Model Class

```python
class Model:
    def __init__(self):
        # Initializes the model
        pass
    
    def add_layer(self, units, activation):
        # Adds a layer to the model
        pass
    
    def compile(self, loss, optimizer):
        # Compiles the model
        pass
    
    def fit(self, X, y, epochs, batch_size):
        # Trains the model on the provided data
        pass
    
    def evaluate(self, X, y):
        # Evaluates the model on the provided data
        pass
```

#### Loss Functions

```python
class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        # Calculates mean squared error
        pass
```

#### Activation Functions

```python
class ReLU:
    def __call__(self, x):
        # Applies ReLU activation function
        pass
```

## Examples

Explore the `examples/` directory for various Jupyter notebooks demonstrating how to use `PyDeepFlow` for different deep learning tasks:

- **Iris_Load_and_Save.py**: Load and save Iris dataset.
- **Iris_Multi_Class.py**: Perform multi-class classification on the Iris dataset.
- **Titanic.ipynb**: Analyze Titanic survival data and build a predictive model.

## References

1. **Neural Networks and Deep Learning**:
   - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Link](https://www.deeplearningbook.org/)

2. **Python for Data Analysis**:
   - McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython*. O'Reilly Media.

3. **Scikit-learn Documentation**:
   - Scikit-learn. (n.d.). *Scikit-learn: Machine Learning in Python*. [Link](https://scikit-learn.org/stable/)

4. **TQDM Documentation**:
   - TQDM. (n.d.). *TQDM: A fast, extensible progress bar for Python and CLI*. [Link](https://tqdm.github.io/)

5. **Colorama Documentation**:
   - Colorama. (n.d.). *Colorama: Simple cross-platform print formatting*. [Link](https://pypi.org/project/colorama/)


## Contributing

We welcome contributions to `PyDeepFlow`. If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

### Guidelines

- Ensure that your code adheres to the existing code style.
- Include tests for any new functionality.
- Update documentation if necessary.

## License

`PyDeepFlow` is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or inquiries, please contact:

- **Author**: Ravin D
- **Email**: [ravin.d3107@outlook.com](mailto:ravin.d3107@outlook.com)
- **GitHub**: [GitHub Profile](https://github.com/ravin-d-27)


---

