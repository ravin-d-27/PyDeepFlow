# Building an Artificial Neural Network (ANN) with Backpropagation

This README explains the mathematical foundations of an Artificial Neural Network (ANN) with backpropagation. The network consists of an input layer, a hidden layer, and an output layer.

## Network Structure

Let’s define the structure of the ANN:
- **Input Layer**: \( n \) neurons (features from the input data)
- **Hidden Layer**: \( h \) neurons
- **Output Layer**: \( m \) neurons (in this example, 1 for binary classification)

---

## 1. Forward Propagation

Forward propagation involves calculating the activations of neurons in the hidden and output layers based on the input data and weights.

### 1.1 Input to Hidden Layer

The input to the hidden layer is the weighted sum of the input features plus the bias:

\[
Z^{(1)} = X \cdot W^{(1)} + b^{(1)}
\]

Where:
- \( X \in \mathbb{R}^{N \times n} \): Input data (with \( N \) samples and \( n \) features)
- \( W^{(1)} \in \mathbb{R}^{n \times h} \): Weight matrix between the input layer and hidden layer
- \( b^{(1)} \in \mathbb{R}^{1 \times h} \): Bias vector for the hidden layer

Next, we apply the **sigmoid activation function** to get the activation of the hidden layer:

\[
A^{(1)} = \sigma(Z^{(1)}) = \frac{1}{1 + e^{-Z^{(1)}}}
\]

Where:
- \( A^{(1)} \in \mathbb{R}^{N \times h} \): The activations (outputs) of the hidden layer
- \( \sigma(x) \): Sigmoid function, defined as \( \sigma(x) = \frac{1}{1 + e^{-x}} \)

### 1.2 Hidden to Output Layer

Similarly, the input to the output layer is computed as a weighted sum of the hidden layer activations plus the bias:

\[
Z^{(2)} = A^{(1)} \cdot W^{(2)} + b^{(2)}
\]

Where:
- \( A^{(1)} \in \mathbb{R}^{N \times h} \): Activations from the hidden layer
- \( W^{(2)} \in \mathbb{R}^{h \times m} \): Weight matrix between hidden and output layer
- \( b^{(2)} \in \mathbb{R}^{1 \times m} \): Bias vector for the output layer

Finally, apply the sigmoid function to get the output of the network:

\[
A^{(2)} = \sigma(Z^{(2)}) = \frac{1}{1 + e^{-Z^{(2)}}}
\]

Where:
- \( A^{(2)} \in \mathbb{R}^{N \times m} \): Predicted output (for binary classification, this would be the probability of class 1)

---

## 2. Loss Function

The **Mean Squared Error (MSE)** loss function is used to calculate the error between predicted and actual values:

\[
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2
\]

Where:
- \( y_i \): Actual output for sample \( i \)
- \( \hat{y}_i \): Predicted output for sample \( i \)
- \( N \): Total number of samples

---

## 3. Backpropagation

Backpropagation is used to update the weights and biases in the network by calculating the gradient of the loss function with respect to the weights.

### 3.1 Output Layer to Hidden Layer

The error at the output layer is computed as the difference between the predicted output and the true output:

\[
\delta^{(2)} = A^{(2)} - y
\]

The gradient of the loss with respect to the weights in the output layer is given by:

\[
\frac{\partial \text{Loss}}{\partial W^{(2)}} = A^{(1)^T} \cdot \delta^{(2)}
\]

The bias gradient for the output layer is:

\[
\frac{\partial \text{Loss}}{\partial b^{(2)}} = \sum_{i=1}^{N} \delta^{(2)}_i
\]

### 3.2 Hidden Layer to Input Layer

Next, compute the error at the hidden layer:

\[
\delta^{(1)} = \left( \delta^{(2)} \cdot W^{(2)^T} \right) \odot \sigma'(Z^{(1)})
\]

Where:
- \( \sigma'(Z^{(1)}) \) is the derivative of the sigmoid function with respect to \( Z^{(1)} \), given by:
  \[
  \sigma'(Z^{(1)}) = A^{(1)} \odot (1 - A^{(1)})
  \]

The gradient of the loss with respect to the weights in the hidden layer is:

\[
\frac{\partial \text{Loss}}{\partial W^{(1)}} = X^T \cdot \delta^{(1)}
\]

The bias gradient for the hidden layer is:

\[
\frac{\partial \text{Loss}}{\partial b^{(1)}} = \sum_{i=1}^{N} \delta^{(1)}_i
\]

---

## 4. Weight and Bias Updates

The weights and biases are updated using gradient descent:

\[
W^{(2)} = W^{(2)} - \alpha \frac{\partial \text{Loss}}{\partial W^{(2)}}
\]

\[
b^{(2)} = b^{(2)} - \alpha \frac{\partial \text{Loss}}{\partial b^{(2)}}
\]

\[
W^{(1)} = W^{(1)} - \alpha \frac{\partial \text{Loss}}{\partial W^{(1)}}
\]

\[
b^{(1)} = b^{(1)} - \alpha \frac{\partial \text{Loss}}{\partial b^{(1)}}
\]

Where:
- \( \alpha \) is the learning rate.

---

## 5. Sigmoid Derivative

The derivative of the sigmoid function \( \sigma(x) \) is needed for backpropagation:

\[
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
\]

This is used to compute the gradients for updating the weights and biases during backpropagation.

---

## Summary of Equations

- **Forward Propagation**:
  \[
  Z^{(1)} = X \cdot W^{(1)} + b^{(1)}, \quad A^{(1)} = \sigma(Z^{(1)})
  \]
  \[
  Z^{(2)} = A^{(1)} \cdot W^{(2)} + b^{(2)}, \quad A^{(2)} = \sigma(Z^{(2)})
  \]

- **Loss Function**:
  \[
  \text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2
  \]

- **Backpropagation**:
  \[
  \delta^{(2)} = A^{(2)} - y
  \]
  \[
  \delta^{(1)} = \left( \delta^{(2)} \cdot W^{(2)^T} \right) \odot \sigma'(Z^{(1)})
  \]

- **Weight Updates**:
  \[
  W^{(2)} = W^{(2)} - \alpha \frac{\partial \text{Loss}}{\partial W^{(2)}}, \quad W^{(1)} = W^{(1)} - \alpha \frac{\partial \text{Loss}}{\partial W^{(1)}}
  \]
  \[
  b^{(2)} = b^{(2)} - \alpha \frac{\partial \text{Loss}}{\partial b^{(2)}}, \quad b^{(1)} = b^{(1)} - \alpha \frac{\partial \text{Loss}}{\partial b^{(1)}}
  \]
