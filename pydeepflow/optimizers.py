import numpy as np


class Adam:
    """
    Adam optimizer.

    Adam is an optimization algorithm that can be used instead of the classical
    stochastic gradient descent procedure to update network weights iteratively
    based in training data.

    Args:
        learning_rate (float): The learning rate.
        beta1 (float): The exponential decay rate for the first moment estimates.
        beta2 (float): The exponential decay rate for the second-moment estimates.
        epsilon (float): A small constant for numerical stability.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        """
        Update parameters.

        Args:
            params (list): List of parameters to update.
            grads (list): List of gradients for each parameter.
        """
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop:
    """
    RMSprop optimizer.

    RMSprop is an unpublished, adaptive learning rate method. The original paper
    is available at http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.

    Args:
        learning_rate (float): The learning rate.
        decay_rate (float): The decay rate.
        epsilon (float): A small constant for numerical stability.
    """

    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        """
        Update parameters.

        Args:
            params (list): List of parameters to update.
            grads (list): List of gradients for each parameter.
        """
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * (
                grads[i] ** 2
            )
            params[i] -= (
                self.learning_rate * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)
            )
