import numpy as np
import matplotlib.pyplot as plt
from pydeepflow.activations import activation, activation_derivative
from pydeepflow.losses import get_loss_function, get_loss_derivative
from pydeepflow.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pydeepflow.device import Device
from pydeepflow.regularization import Regularization
from pydeepflow.checkpoints import ModelCheckpoint
from pydeepflow.cross_validator import CrossValidator
from pydeepflow.batch_normalization import BatchNormalization
from tqdm import tqdm
from pydeepflow.optimizers import Adam, RMSprop
import time
import sys

# ====================================================================
# IM2COL / COL2IM helper functions
# ====================================================================

def get_im2col_indices(X_shape, filter_height, filter_width, padding=0, stride=1):
    """
    Creates indices to use with fancy indexing to get column matrix from a 4D tensor (N, H, W, C).
    """
    N, H, W, C = X_shape
    Fh, Fw = filter_height, filter_width

    # Calculate output dimensions
    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1
    
    # Check if dimensions are valid (no fractional output size)
    if not isinstance(H_out, int) or not isinstance(W_out, int):
        raise ValueError("Invalid stride or filter size for input dimensions.")

    # Indices of the output grid
    i0 = np.repeat(np.arange(Fh), Fw)
    i1 = np.tile(np.arange(Fw), Fh)
    
    # Indices in the original padded image
    i = i0.reshape(-1, 1) + np.arange(H_out) * stride
    j = i1.reshape(-1, 1) + np.arange(W_out) * stride

    # Indices for all filters, rows, columns, and channels
    k = np.repeat(np.arange(C), Fh * Fw).reshape(-1, 1)
    
    # Repeat and tile for the output feature map size
    i = np.repeat(i, W_out, axis=1)
    j = np.tile(j, H_out).reshape(i.shape)

    return (k, i, j)

def im2col_indices(X, filter_height, filter_width, padding=0, stride=1):
    """
    Transforms the input tensor X into a column matrix.
    Input X shape: (N, H, W, C) -> Output: (N * H_out * W_out, Fh * Fw * C)
    """
    N, H, W, C = X.shape
    Fh, Fw = filter_height, filter_width
    
    # Pad input
    X_padded = X
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')

    # Calculate output dimensions
    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1

    # Create patches using broadcasting
    X_col = np.zeros((N, H_out, W_out, Fh * Fw * C))
    
    for y in range(Fh):
        y_max = y + stride * H_out
        for x in range(Fw):
            x_max = x + stride * W_out
            X_col[:, :, :, y * Fw + x::Fh * Fw] = X_padded[:, y:y_max:stride, x:x_max:stride, :]
    
    # Reshape to (N * H_out * W_out, Fh * Fw * C)
    X_col = X_col.reshape(N * H_out * W_out, -1)
    return X_col


def col2im_indices(cols, X_shape, filter_height, filter_width, padding=0, stride=1):
    """
    Inverse of im2col. Converts columns back to the image shape. Returns shape (N, H, W, C).
    """
    N, H, W, C = X_shape
    Fh, Fw = filter_height, filter_width

    H_out = (H + 2 * padding - Fh) // stride + 1
    W_out = (W + 2 * padding - Fw) // stride + 1

    # Initialize gradient container
    if padding > 0:
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        X_grad = np.zeros((N, H_padded, W_padded, C))
    else:
        X_grad = np.zeros(X_shape)

    # Reshape cols back to (N, H_out, W_out, Fh * Fw * C)
    cols_reshaped = cols.reshape(N, H_out, W_out, Fh * Fw * C)

    # Scatter-add contributions back to input positions
    for y in range(Fh):
        y_max = y + stride * H_out
        for x in range(Fw):
            x_max = x + stride * W_out
            if padding > 0:
                X_grad[:, y:y_max:stride, x:x_max:stride, :] += cols_reshaped[:, :, :, y * Fw + x::Fh * Fw]
            else:
                X_grad[:, y:y_max:stride, x:x_max:stride, :] += cols_reshaped[:, :, :, y * Fw + x::Fh * Fw]

    # Remove padding if it was added
    if padding > 0:
        return X_grad[:, padding:-padding, padding:-padding, :]
    return X_grad


# ====================================================================
# ConvLayer and Flatten
# ====================================================================

class ConvLayer:
    """
    2D convolutional layer (channels-last) implemented via im2col.
    - in_channels: number of input channels (C_prev)
    - out_channels: number of filters
    - kernel_size: int (assumes square kernels)
    - stride: stride
    - padding: integer padding amount (same padding when appropriate)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Fh = kernel_size
        self.Fw = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device if device is not None else Device(use_gpu=False)

        # He/Kaiming initialization
        scale = np.sqrt(2.0 / (self.Fh * self.Fw * self.in_channels))
        W_init = self.device.random().randn(self.Fh, self.Fw, self.in_channels, self.out_channels) * scale
        b_init = self.device.zeros((1, 1, 1, self.out_channels))

        # store params in a dict for clarity
        self.params = {'W': W_init, 'b': b_init}
        self.grads = {'dW': None, 'db': None}
        self.cache = None

    def forward(self, X):
        """
        X shape: (N, H, W, C_prev)
        Returns: (N, H_out, W_out, out_channels)
        """
        N, H, W, C = X.shape
        Fh, Fw = self.Fh, self.Fw

        H_out = (H + 2 * self.padding - Fh) // self.stride + 1
        W_out = (W + 2 * self.padding - Fw) // self.stride + 1

        # Convert input to columns
        X_col = im2col_indices(X, Fh, Fw, self.padding, self.stride)  # (N*H_out*W_out, Fh*Fw*C_prev)

        # Reshape weights to (Fh*Fw*C_prev, out_channels)
        W_col = self.params['W'].reshape(-1, self.out_channels)

        # GEMM: (N*H_out*W_out, out_channels)
        out_col = self.device.dot(X_col, W_col)

        # reshape back to image format
        out = out_col.reshape(N, H_out, W_out, self.out_channels)

        # add bias (broadcasts over N, H_out, W_out)
        out = out + self.params['b']

        # cache for backward pass
        self.cache = (X.shape, X_col, self.params['W'], self.params['b'])

        return out

    def backward(self, dOut):
        """
        dOut shape: (N, H_out, W_out, out_channels)
        Returns gradient wrt input X with shape (N, H, W, C_prev)
        Also populates self.grads['dW'] and self.grads['db'].
        """
        X_shape, X_col, W, b = self.cache
        N, H, W_in, C = X_shape  # W_in variable name reused; keep clear variable naming below

        # flatten dOut to columns: (N*H_out*W_out, out_channels)
        dOut_col = dOut.reshape(-1, self.out_channels)

        # bias gradient: sum over batch and spatial dims -> shape (1,1,1,out_channels)
        db = self.device.sum(dOut, axis=(0, 1, 2), keepdims=True)

        # weight gradient: X_col.T @ dOut_col -> (Fh*Fw*C_prev, out_channels)
        dW_col = self.device.dot(X_col.T, dOut_col)
        dW = dW_col.reshape(self.Fh, self.Fw, self.in_channels, self.out_channels)

        # store grads correctly
        self.grads['dW'] = dW
        self.grads['db'] = db

        # input gradient: dX_col = dOut_col @ W_col.T
        W_col = W.reshape(-1, self.out_channels)
        dX_col = self.device.dot(dOut_col, W_col.T)

        # convert columns back to image gradient
        dX = col2im_indices(dX_col, X_shape, self.Fh, self.Fw, self.padding, self.stride)

        return dX


class Flatten:
    """
    Flattens (N, H, W, C) -> (N, H*W*C)
    """
    def __init__(self):
        self.cache = None
        self.params = {}
        self.grads = {}

    def forward(self, X):
        self.cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dOut):
        original_shape = self.cache
        return dOut.reshape(original_shape)


# ====================================================================
# Multi_Layer_ANN (Dense-only training logic) - cleaned up
# ====================================================================

class Multi_Layer_ANN:
    """
    A Multi-Layer Dense ANN wrapper (keeps compatibility with earlier code).
    Note: This class currently handles Dense layers only. Conv/Flatten must be
    integrated into a higher-level 'Sequential' class for end-to-end training.
    """
    def __init__(self, X_train, Y_train, hidden_layers, activations, loss='categorical_crossentropy',
                 use_gpu=False, l2_lambda=0.0, dropout_rate=0.0, use_batch_norm=False, optimizer='sgd'):
        self.device = Device(use_gpu=use_gpu)
        self.regularization = Regularization(l2_lambda, dropout_rate)

        # internal dense parameters
        self.dense_layer_weights = []
        self.dense_layer_biases = []

        # determine architecture
        if Y_train.ndim == 1 or (Y_train.ndim > 1 and Y_train.shape[1] == 1):
            self.layers = [X_train.shape[1]] + hidden_layers + [1]
            self.output_activation = 'sigmoid'
        else:
            self.layers = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1]]
            self.output_activation = 'softmax'

        self.activations = activations

        # external lists kept for compatibility
        self.weights = []
        self.biases = []

        if len(self.activations) != len(hidden_layers):
            raise ValueError("The number of activation functions must match the number of hidden layers.")

        # loss setup
        self.loss = loss
        self.loss_func = get_loss_function(self.loss)
        self.loss_derivative = get_loss_derivative(self.loss)

        # move data to device (wrapper)
        self.X_train = self.device.array(X_train)
        self.y_train = self.device.array(Y_train)

        # initialize dense weights (He init)
        for i in range(len(self.layers) - 1):
            w = self.device.random().randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / max(1, self.layers[i]))
            b = self.device.zeros((1, self.layers[i + 1]))
            self.dense_layer_weights.append(w)
            self.dense_layer_biases.append(b)

        # sync external lists
        self.weights = self.dense_layer_weights
        self.biases = self.dense_layer_biases

        self.training = False

        # history initialization
        self.history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        # batch norm
        self.use_batch_norm = use_batch_norm
        self.batch_norm_layers = []
        if self.use_batch_norm:
            for i in range(len(self.layers) - 2):
                self.batch_norm_layers.append(BatchNormalization(self.layers[i + 1], device=self.device))

        # optimizer selection
        if optimizer == 'adam':
            self.optimizer = Adam()
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop()
        else:
            self.optimizer = None

    def forward_propagation(self, X):
        activations = [X]
        Z_values = []

        for i in range(len(self.dense_layer_weights) - 1):
            Z = self.device.dot(activations[-1], self.dense_layer_weights[i]) + self.dense_layer_biases[i]
            if self.use_batch_norm:
                Z = self.batch_norm_layers[i].normalize(Z, training=self.training)
            Z_values.append(Z)
            A = activation(Z, self.activations[i], self.device)
            A = self.regularization.apply_dropout(A, training=self.training)
            activations.append(A)

        # output layer
        Z_output = self.device.dot(activations[-1], self.dense_layer_weights[-1]) + self.dense_layer_biases[-1]
        A_output = activation(Z_output, self.output_activation, self.device)
        Z_values.append(Z_output)
        activations.append(A_output)

        return activations, Z_values

    def backpropagation(self, X, y, activations, Z_values, learning_rate, clip_value=None):
        # output layer error
        output_error = activations[-1] - y
        d_output = output_error * activation_derivative(activations[-1], self.output_activation, self.device)

        # backpropagate through hidden layers
        deltas = [d_output]
        for i in reversed(range(len(self.dense_layer_weights) - 1)):
            error = self.device.dot(deltas[-1], self.dense_layer_weights[i + 1].T)
            if self.use_batch_norm:
                error = self.batch_norm_layers[i].backprop(Z_values[i], error, learning_rate)
            delta = error * activation_derivative(activations[i + 1], self.activations[i], self.device)
            deltas.append(delta)

        deltas.reverse()

        # compute gradients
        gradient = {'weights': [], 'biases': []}
        for i in range(len(self.dense_layer_weights)):
            grad_weights = self.device.dot(activations[i].T, deltas[i])
            grad_biases = self.device.sum(deltas[i], axis=0, keepdims=True)

            if clip_value is not None:
                grad_weights_norm = self.device.norm(grad_weights)
                if grad_weights_norm > clip_value:
                    grad_weights = grad_weights * (clip_value / grad_weights_norm)

                grad_biases_norm = self.device.norm(grad_biases)
                if grad_biases_norm > clip_value:
                    grad_biases = grad_biases * (clip_value / grad_biases_norm)

            gradient['weights'].append(grad_weights)
            gradient['biases'].append(grad_biases)

        # parameter update
        if self.optimizer:
            all_params = self.dense_layer_weights + self.dense_layer_biases
            all_grads = gradient['weights'] + gradient['biases']
            self.optimizer.update(all_params, all_grads)
            self.dense_layer_weights = all_params[:len(self.dense_layer_weights)]
            self.dense_layer_biases = all_params[len(self.dense_layer_weights):]
        else:
            for i in range(len(self.dense_layer_weights)):
                self.dense_layer_weights[i] -= gradient['weights'][i] * learning_rate
                self.dense_layer_biases[i] -= gradient['biases'][i] * learning_rate

        # L2 regularization (apply to weights only)
        for i in range(len(self.dense_layer_weights)):
            self.dense_layer_weights[i] -= learning_rate * self.regularization.apply_l2_regularization(self.dense_layer_weights[i], learning_rate, X.shape)

        # sync for compatibility
        self.weights = self.dense_layer_weights
        self.biases = self.dense_layer_biases

    def fit(self, epochs, learning_rate=0.01, lr_scheduler=None, early_stop=None, X_val=None, y_val=None, checkpoint=None, verbose=False, clipping_threshold=None):
        self.weights = self.dense_layer_weights
        self.biases = self.dense_layer_biases

        if early_stop:
            assert X_val is not None and y_val is not None, "Validation set required for early stopping"

        for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii="░▒█", colour='green', disable=not verbose):
            start_time = time.time()

            if lr_scheduler is not None:
                current_lr = lr_scheduler.get_lr(epoch)
            else:
                current_lr = learning_rate

            self.training = True
            activations, Z_values = self.forward_propagation(self.X_train)
            self.backpropagation(self.X_train, self.y_train, activations, Z_values, current_lr, clip_value=clipping_threshold)
            self.training = False

            # metrics
            train_loss = self.loss_func(self.y_train, activations[-1], self.device)
            train_accuracy = np.mean((activations[-1] >= 0.5).astype(int) == self.y_train) if self.output_activation == 'sigmoid' else np.mean(np.argmax(activations[-1], axis=1) == np.argmax(self.y_train, axis=1))

            if train_loss is None or train_accuracy is None:
                print("Warning: train_loss or train_accuracy is None!")
                continue

            val_loss = val_accuracy = None
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(self.device.array(X_val))
                val_loss = self.loss_func(self.device.array(y_val), val_activations[-1], self.device)
                val_accuracy = np.mean((val_activations[-1] >= 0.5).astype(int) == y_val) if self.output_activation == 'sigmoid' else np.mean(np.argmax(val_activations[-1], axis=1) == np.argmax(y_val, axis=1))

            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

            if checkpoint is not None and X_val is not None:
                if checkpoint.should_save(epoch, val_loss):
                    checkpoint.save_weights(epoch, self.weights, self.biases, val_loss)

            if verbose and (epoch % 10 == 0):
                sys.stdout.write(
                    f"\rEpoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Accuracy: {train_accuracy:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Accuracy: {val_accuracy:.4f} | "
                    f"LR: {current_lr:.6f}  "
                )
                sys.stdout.flush()

            if early_stop:
                early_stop(val_loss)
                if early_stop.early_stop:
                    print('\n', "#" * 80)
                    print(f"Early stop at epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
                    print('#' * 80)
                    break

        print("Training Completed!")

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def evaluate(self, X, y, metrics=['loss', 'accuracy']):
        predictions = self.predict(X)
        results = {}

        if 'loss' in metrics:
            results['loss'] = self.loss_func(y, predictions, self.device)

        y_pred_classes = (predictions >= 0.5).astype(int) if self.output_activation == 'sigmoid' else np.argmax(predictions, axis=1)
        y_true_classes = y if self.output_activation == 'sigmoid' else np.argmax(y, axis=1)

        if 'accuracy' in metrics:
            results['accuracy'] = np.mean(y_pred_classes == y_true_classes)

        if 'precision' in metrics:
            results['precision'] = precision_score(y_true_classes, y_pred_classes)

        if 'recall' in metrics:
            results['recall'] = recall_score(y_true_classes, y_pred_classes)

        if 'f1_score' in metrics:
            results['f1_score'] = f1_score(y_true_classes, y_pred_classes)

        if 'confusion_matrix' in metrics:
            num_classes = self.layers[-1]
            results['confusion_matrix'] = confusion_matrix(y_true_classes, y_pred_classes, num_classes)

        return results

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = ModelCheckpoint(checkpoint_path)
        self.weights, self.biases = checkpoint.load_weights()
        self.dense_layer_weights = self.weights
        self.dense_layer_biases = self.biases

    def save_model(self, file_path):
        self.weights = self.dense_layer_weights
        self.biases = self.dense_layer_biases

        model_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'layers': self.layers,
            'activations': self.activations,
            'output_activation': self.output_activation
        }
        np.save(file_path, model_data)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        model_data = np.load(file_path, allow_pickle=True).item()
        self.weights = [self.device.array(w) for w in model_data['weights']]
        self.biases = [self.device.array(b) for b in model_data['biases']]
        self.layers = model_data['layers']
        self.activations = model_data['activations']
        self.output_activation = model_data['output_activation']
        print(f"Model loaded from {file_path}")
        self.dense_layer_weights = self.weights
        self.dense_layer_biases = self.biases


# ====================================================================
# Plotting utilities
# ====================================================================

class Plotting_Utils:
    def plot_training_history(self, history, metrics=('loss', 'accuracy'), figure='history.png'):
        num_metrics = len(metrics)
        fig, ax = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))

        if num_metrics == 1:
            ax = [ax]

        for i, metric in enumerate(metrics):
            ax[i].plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
            if f'val_{metric}' in history:
                ax[i].plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
            ax[i].set_title(f"{metric.capitalize()} over Epochs")
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(metric.capitalize())
            ax[i].legend()

        plt.savefig(figure)
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, train_sizes, train_scores, val_scores, figure='learning_curve.png'):
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                             val_scores_mean + val_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
                     label="Cross-validation score")

        plt.legend(loc="best")
        plt.savefig(figure)
        plt.show()
