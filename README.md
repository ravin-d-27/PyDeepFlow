
---

# PyDeepFlow

### Author & Creator: **ravin-d-27**

---

## Overview

This documentation covers the development, structure, and features of PyDeepFlow created for Deep Learning workflows (Right Now, there is only support for Multi-Class Classification). The model is implemented from scratch using Python and NumPy and offers flexibility in terms of architecture, activation functions, and training methods. Future enhancement plans are also outlined, showcasing how the model can evolve to meet more complex and diverse requirements.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Model Architecture](#model-architecture)
3. [Implementation Details](#implementation-details)
4. [Features and Functionality](#features-and-functionality)
5. [Future Enhancement Plans](#future-enhancement-plans)
6. [How to Use the Model](#how-to-use-the-model)
7. [Example Code](#example-code)
8. [Conclusion](#conclusion)

---

## 1. Introduction

This custom Multi-Layer Artificial Neural Network was built by **ravin-d-27** to solve binary classification problems. The model is designed to be easy to use, extendable, and adaptable. With this custom-built architecture, users can define the number of layers, the activation functions, and control the learning process. Additionally, the ANN supports various loss functions, allowing flexibility depending on the specific problem at hand.

While the current version focuses on binary classification, several enhancements are planned to expand its capabilities, making it a more comprehensive neural network tool for both binary and multi-class classification tasks.

---

## 2. Model Architecture

The architecture of the neural network consists of several layers:

### **Input Layer**:
- The number of neurons in the input layer is automatically determined by the number of features in the training data.
  
### **Hidden Layers**:
- The number of hidden layers and neurons per layer is configurable.
- The model supports different activation functions for each layer, making it flexible to different types of tasks.

### **Output Layer**:
- A single neuron with a sigmoid activation function is used in the output layer to output a probability score for binary classification (0 or 1).

The forward pass includes activation functions like ReLU, Leaky ReLU, Sigmoid, and Tanh. The backpropagation uses derivatives of these activation functions to update the weights during training.

### **Activation Functions Implemented**:
- **ReLU** (Rectified Linear Unit)
- **Leaky ReLU**
- **Sigmoid**
- **Tanh**
- **Softmax** (planned for multi-class tasks)

---

## 3. Implementation Details

The ANN is structured into three core modules:

### **activations.py**:
This file includes a set of commonly used activation functions and their derivatives, which are essential for forward propagation and backpropagation. The functions implemented are:

- **ReLU** and its derivative
- **Leaky ReLU** and its derivative
- **Sigmoid** and its derivative
- **Tanh** and its derivative
- **Softmax** (though only used for future multi-class implementations)

### **losses.py**:
The `losses.py` module handles different types of loss functions. It allows users to select from multiple loss functions based on their task requirements. Available loss functions include:

- **Binary Crossentropy**: Used for binary classification.
- **Mean Squared Error (MSE)**: Useful for regression tasks.
- **Categorical Crossentropy**: Will be needed for multi-class classification.
- **Hinge Loss**: Typically used for support vector machines (SVM).
- **Huber Loss**: A hybrid loss function that is less sensitive to outliers.

Each loss function has its corresponding derivative, which is essential for backpropagation.

### **model.py**:
The `model.py` file contains the core **Multi_Layer_ANN** class, which implements:
- **Weight Initialization**: Weights are initialized using a He initialization method, which scales the weights relative to the number of neurons in each layer to avoid exploding/vanishing gradients.
- **Forward Propagation**: Handles the feed-forward phase of the network.
- **Backpropagation**: Computes the gradients of the loss with respect to weights and biases using the chain rule.
- **Training Loop**: Manages the optimization process over a set number of epochs, adjusting weights and biases based on the calculated gradients.
- **Prediction Method**: Outputs predictions after training by thresholding the sigmoid output for binary classification.

#### Training Loop Details
The training loop provides feedback on the loss and accuracy at regular intervals, making it easy to monitor performance. The model is trained using stochastic gradient descent (SGD) with a customizable learning rate. During training, the model:
- Executes forward propagation.
- Computes the loss using the specified loss function.
- Performs backpropagation to adjust weights and biases.
- Repeats this process over the defined number of epochs.

---

## 4. Features and Functionality

### **Core Features**:
- **Configurable Hidden Layers**: The architecture can include any number of hidden layers with varying neuron counts and activation functions.
- **Binary Classification Support**: The current model is built for binary classification tasks, utilizing sigmoid activation in the output layer.
- **Customizable Loss Functions**: The model allows users to specify different loss functions depending on the task.
- **Training Feedback**: Detailed feedback on loss and accuracy during training, displayed at regular intervals.

### **Model Metrics**:
- **Accuracy**: Calculated based on how well the model's predictions match the actual labels.
- **Loss**: Calculated using the chosen loss function, guiding the optimization process.

---

## 5. Future Enhancement Plans

To further extend the model’s functionality, the following enhancements will be added:

1. **Regularization Techniques**:
   - **L2 Regularization**: Penalizes large weights to reduce overfitting.
   - **Dropout**: Randomly disables neurons during training, increasing generalization ability.

2. **Advanced Optimizers**:
   - Support for optimizers like **Adam**, **RMSprop**, and **AdaGrad** to enhance convergence speed and improve performance on complex datasets.

3. **Learning Rate Scheduling**:
   - Dynamic adjustment of the learning rate over time (e.g., **learning rate decay** or **cyclic learning rates**) to improve training stability.

4. **Early Stopping**:
   - Stop training when there is no significant improvement in validation loss to prevent overfitting and reduce unnecessary computation.

5. ~~**Support for Multi-Class Classification**:~~
   - ~~Modify the output layer and implement softmax activation for tasks requiring multi-class predictions.~~

6. **Model Checkpointing**:
   - Save the model’s weights at optimal points during training to prevent loss of progress, especially for long training times.

7. **Batch Normalization**:
   - Add batch normalization layers between the hidden layers to stabilize and speed up training.

8. **Gradient Clipping**:
   - Prevent exploding gradients by limiting the magnitude of gradient updates during backpropagation.

9. **Visualization Tools**:
   - Introduce functions for plotting training metrics (loss, accuracy) over time for better tracking and debugging.

10. **Hyperparameter Tuning Framework**:
    - Implement support for **grid search** or **random search** to allow for efficient exploration of hyperparameter combinations.

11. **Additional Activation Functions**:
    - Add support for more advanced activation functions like **Swish** and **ELU** for better performance on deeper networks.

12. **Cross-Validation Support**:
    - Add support for k-fold cross-validation to ensure the model generalizes well across different subsets of data.

13. **Dynamic Model Architecture**:
    - Allow for more flexibility in configuring the architecture dynamically by specifying activation functions, number of layers, and neuron counts directly.

14. **Support for Convolutional Layers**:
    - Add support for convolutional layers (CNN) to extend the model for image data processing.

15. **Output Probabilities in Predictions**:
    - Modify the `predict()` method to return not just binary labels but also the predicted probability scores, providing more interpretability.

---

## 6. How to Use the Model

### Prerequisites:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- tqdm (for progress bars during training)
- colorama (for colored console outputs)

### Steps to Use:
1. **Prepare Data**:
   - The input data should be a 2D array where rows represent examples and columns represent features.
   - The target (label) should be a binary value (0 or 1) for binary classification.

2. **Split Data**:
   - Use `train_test_split` to divide your dataset into training and testing sets.

3. **Standardize Data**:
   - Scale the features using **StandardScaler** to normalize the input data.

4. **Initialize the ANN**:
   - Specify the architecture by defining the hidden layers and the activation functions.

5. **Train the Model**:
   - Train the model by calling the `fit()` method with the desired number of epochs and learning rate.

6. **Make Predictions**:
   - Use the `predict()` method to classify new data after training.

---

## 7. Example Code

```python
   X = df.iloc[:, :-1].values
   y = df.iloc[:, -1].values

   y_one_hot = np.eye(len(np.unique(y)))[y]

   X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)


    # Define the architecture
   hidden_layers = [5, 5]
   activations = ['relu', 'relu']

   ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, loss='categorical_crossentropy')
   ann.fit(epochs=1000, learning_rate=0.01)

    # Make predictions
   y_pred = ann.predict(X_test)

   print(y_pred)

    # Convert predictions back to original labels
   y_test_labels = np.argmax(y_test, axis=1)

    # Calculate accuracy
   accuracy = np.mean(y_pred == y_test_labels)
   print(f"Test Accuracy: {accuracy * 100:.2f}%")

```

---

## 9. References

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

---

## 10. Contributions

Contributions to this project are welcome! Here’s how you can contribute:

1. **Fork the Repository**: Make a personal copy of the repository.
2. **Create a Feature Branch**: Use a descriptive name for the branch that outlines the feature being added (e.g., `feature/dropout`).
3. **Make Your Changes**: Implement the changes you wish to contribute.
4. **Commit Your Changes**: Write clear, concise commit messages.
5. **Push to Your Fork**: Push your changes back to your personal fork of the repository.
6. **Open a Pull Request**: Describe the changes and the reasoning behind them.

### Issues
If you encounter any bugs or have suggestions for improvement, please open an issue in the repository to discuss it!

---

## 11. Acknowledgments

- **Inspirations**: This project is inspired by numerous resources available in the machine learning community, particularly literature on neural networks and deep learning.
- **Community Support**: Thanks to the contributors and open-source community for their valuable insights and discussions that shaped this project.

---

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 13. Contact

For any inquiries or suggestions regarding this project, please feel free to contact me:
- **GitHub**: [ravin-d-27](https://github.com/ravin-d-27)

---
