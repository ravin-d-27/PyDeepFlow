import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

class Device:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if use_gpu and cp is None:
            raise ValueError("CuPy is not installed, please install CuPy for GPU support.")

    def array(self, data):
        return cp.array(data) if self.use_gpu else np.array(data)

    def zeros(self, shape):
        return cp.zeros(shape) if self.use_gpu else np.zeros(shape)

    def random(self):
        return cp.random if self.use_gpu else np.random

    def exp(self, x):
        return cp.exp(x) if self.use_gpu else np.exp(x)

    def dot(self, a, b):
        return cp.dot(a, b) if self.use_gpu else np.dot(a, b)

    def maximum(self, a, b):
        return cp.maximum(a, b) if self.use_gpu else np.maximum(a, b)

    def tanh(self, x):
        return cp.tanh(x) if self.use_gpu else np.tanh(x)

    def sum(self, x, axis=None, keepdims=False):
        return cp.sum(x, axis=axis, keepdims=keepdims) if self.use_gpu else np.sum(x, axis=axis, keepdims=keepdims)

    def where(self, condition, x, y):
        return cp.where(condition, x, y) if self.use_gpu else np.where(condition, x, y)

    def sqrt(self, x):
        return cp.sqrt(x) if self.use_gpu else np.sqrt(x)

    def log(self, x):
        return cp.log(x) if self.use_gpu else np.log(x)
    
    def asnumpy(self, x):
        return cp.asnumpy(x) if self.use_gpu else x

    # Adding missing max function
    def max(self, x, axis=None, keepdims=False):
        return cp.max(x, axis=axis, keepdims=keepdims) if self.use_gpu else np.max(x, axis=axis, keepdims=keepdims)

