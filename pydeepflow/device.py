import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

class Device:
    """
    A utility class to handle computations on either CPU (NumPy) or GPU (CuPy) 
    depending on the user's preference.

    Parameters:
    -----------
    use_gpu : bool, optional (default=False)
        If True, the class uses GPU (CuPy) for array operations. If CuPy is not installed, 
        it raises an ImportError.
    
    Attributes:
    -----------
    use_gpu : bool
        Whether to use GPU (CuPy) or CPU (NumPy) for computations.

    Raises:
    -------
    ValueError
        If `use_gpu=True` but CuPy is not installed.
    """
    
    def __init__(self, use_gpu=False):
        if use_gpu and not CUPY_AVAILABLE:
            print("Warning: CuPy is not installed. Falling back to CPU.")
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu    

    def abs(self, x):
        """
        Computes the absolute value of each element in the input array.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The absolute value of each element in `x`.
        """
        return cp.abs(x) if self.use_gpu else np.abs(x)
    
    def square(self, x):
        """
        Computes the square of each element in the input array.
    
        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.
    
        Returns:
        --------
        np.ndarray or cp.ndarray
            The element-wise square of the input.
        """
        return cp.square(x) if self.use_gpu else np.square(x)

    def array(self, data):
        """
        Converts input data into a NumPy or CuPy array.

        Parameters:
        -----------
        data : array-like
            The input data to convert.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The converted array, depending on whether GPU or CPU is used.
        """
        return cp.array(data) if self.use_gpu else np.array(data)

    def zeros(self, shape):
        """
        Creates an array of zeros with the specified shape.

        Parameters:
        -----------
        shape : tuple of ints
            The shape of the output array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            An array of zeros, either using NumPy or CuPy.
        """
        return cp.zeros(shape) if self.use_gpu else np.zeros(shape)

    def random(self):
        """
        Returns a random module, either NumPy's or CuPy's, depending on the device.

        Returns:
        --------
        module
            Either `np.random` or `cp.random`.
        """
        return cp.random if self.use_gpu else np.random

    def sign(self, x):
        """
        Computes the sign of each element in the input array.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The sign of each element in `x`.
        """
        return cp.sign(x) if self.use_gpu else np.sign(x)

    def exp(self, x):
        """
        Computes the element-wise exponential of the input array.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The element-wise exponential of the input.
        """
        return cp.exp(x) if self.use_gpu else np.exp(x)

    def dot(self, a, b):
        """
        Computes the dot product of two arrays.

        Parameters:
        -----------
        a : np.ndarray or cp.ndarray
            First input array.
        b : np.ndarray or cp.ndarray
            Second input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The dot product of `a` and `b`.
        """
        return cp.dot(a, b) if self.use_gpu else np.dot(a, b)

    def maximum(self, a, b):
        """
        Element-wise maximum of two arrays.

        Parameters:
        -----------
        a : np.ndarray or cp.ndarray
            First input array.
        b : np.ndarray or cp.ndarray
            Second input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The element-wise maximum of `a` and `b`.
        """
        return cp.maximum(a, b) if self.use_gpu else np.maximum(a, b)

    def tanh(self, x):
        """
        Computes the hyperbolic tangent of the input array.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The hyperbolic tangent of the input.
        """
        return cp.tanh(x) if self.use_gpu else np.tanh(x)

    def sum(self, x, axis=None, keepdims=False):
        """
        Sums the elements of an array along a specified axis.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            Input array.
        axis : int or None, optional (default=None)
            Axis along which the sum is performed.
        keepdims : bool, optional (default=False)
            Whether to keep the reduced dimensions.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The sum of elements in `x` along the specified axis.
        """
        return cp.sum(x, axis=axis, keepdims=keepdims) if self.use_gpu else np.sum(x, axis=axis, keepdims=keepdims)

    def where(self, condition, x, y):
        """
        Return elements chosen from `x` or `y` depending on `condition`.

        Parameters:
        -----------
        condition : array-like
            Where True, yield `x`, otherwise yield `y`.
        x : array-like
            Values from which to choose where `condition` is True.
        y : array-like
            Values from which to choose where `condition` is False.

        Returns:
        --------
        np.ndarray or cp.ndarray
            Array formed by elements from `x` or `y`, depending on the condition.
        """
        return cp.where(condition, x, y) if self.use_gpu else np.where(condition, x, y)

    def sqrt(self, x):
        """
        Computes the square root of the input array, element-wise.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The square root of each element in `x`.
        """
        return cp.sqrt(x) if self.use_gpu else np.sqrt(x)

    def log(self, x):
        """
        Computes the natural logarithm of the input array, element-wise.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The natural logarithm of each element in `x`.
        """
        return cp.log(x) if self.use_gpu else np.log(x)
    
    def asnumpy(self, x):
        """
        Converts a CuPy array to a NumPy array, or simply returns the input if it is already a NumPy array.

        Parameters:
        -----------
        x : cp.ndarray or np.ndarray
            The input array.

        Returns:
        --------
        np.ndarray
            A NumPy array.
        """
        return cp.asnumpy(x) if self.use_gpu else x

    def max(self, x, axis=None, keepdims=False):
        """
        Returns the maximum of an array or along a specific axis.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            Input array.
        axis : int or None, optional (default=None)
            Axis along which to find the maximum.
        keepdims : bool, optional (default=False)
            Whether to keep the reduced dimensions.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The maximum value(s) in `x` along the specified axis.
        """
        return cp.max(x, axis=axis, keepdims=keepdims) if self.use_gpu else np.max(x, axis=axis, keepdims=keepdims)
    
    def norm(self, x, ord=None, axis=None, keepdims=False):
        """
        Matrix or vector norm.

        This function is able to return one of eight different matrix norms,
        or one of an infinite number of vector norms (described below), depending
        on the value of the ``ord`` parameter.

        Parameters
        ----------
        x : np.ndarray or cp.ndarray
            Input array.
        ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional (default=None)
            Order of the norm.
        axis :int or None, optional (default=None)
            Axis along which to find the norm.
        keepdims : bool, optional (default=False)
            Whether to keep the reduced dimensions.

        Returns:
        --------
        float or np.ndarray or cp.ndarray
            Norm of matrix or vector(s).
        """

        return cp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims) if self.use_gpu else np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def ones(self, shape):
        """
        Creates an array of ones with the specified shape.

        Parameters:
        -----------
        shape : tuple of ints
            The shape of the output array.

        Returns:
        --------
        np.ndarray or cp.ndarray
            An array of ones, either using NumPy or CuPy.
        """
        return cp.ones(shape) if self.use_gpu else np.ones(shape)
    
    
    def mean(self, x, axis=None, keepdims=False):
        """
        Computes the mean of the input array along the specified axis.

        Parameters:
        -----------
        x : np.ndarray or cp.ndarray
            The input array.
        axis : int or tuple of ints, optional
            Axis or axes along which the means are computed.
        keepdims : bool, optional
            If True, the reduced dimensions are retained.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The mean of the input array along the specified axis.
        """
        return cp.mean(x, axis=axis, keepdims=keepdims) if self.use_gpu else np.mean(x, axis=axis, keepdims=keepdims)

    def var(self, x, axis=None, keepdims=False):
        """
        Computes the variance of an array along a specified axis.

        Parameters:
        ----------- 
        x : np.ndarray or cp.ndarray
            Input array.
        axis : int or None, optional (default=None)
            Axis along which the variance is computed.
        keepdims : bool, optional (default=False)
            If True, the reduced dimensions will be retained.

        Returns:
        --------
        np.ndarray or cp.ndarray
            The variance of the input array along the specified axis.
        """
        return cp.var(x, axis=axis, keepdims=keepdims) if self.use_gpu else np.var(x, axis=axis, keepdims=keepdims)
