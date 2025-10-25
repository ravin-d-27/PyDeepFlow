import numpy as np
from scipy.ndimage import rotate, shift, zoom


class ImageDataGenerator:
    """
    Generates batches of tensor image data with real-time data augmentation.
    Assumes image format is (height, width, channels).
    """

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False):
        """
        Initializes the ImageDataGenerator.

        Args:
            rotation_range (float): Range (in degrees, 0 to 180) for random rotations.
            width_shift_range (float): Fraction of total width (0 to 1) for random horizontal shifts.
            height_shift_range (float): Fraction of total height (0 to 1) for random vertical shifts.
            zoom_range (float or tuple/list): Range for random zoom.
                - If float, zoom will be [1-zoom_range, 1+zoom_range].
                - If tuple/list [lower, upper], zoom will be [lower, upper].
            horizontal_flip (bool): Whether to randomly flip inputs horizontally.
            vertical_flip (bool): Whether to randomly flip inputs vertically.
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range

        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError("`zoom_range` should be a float or "
                             "a tuple or list of two floats. "
                             f"Received: {zoom_range}")

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def _apply_random_transform(self, x):
        """Applies a random transformation to a single image x."""
        img_h, img_w, img_c = x.shape

        # Rotation
        if self.rotation_range > 0:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            x = rotate(x, theta, reshape=False, order=1, mode='constant', cval=0.)

        # Height Shift
        if self.height_shift_range > 0:
            ty = np.random.uniform(-self.height_shift_range, self.height_shift_range) * img_h
            x = shift(x, (ty, 0, 0), order=1, mode='constant', cval=0.)

        # Width Shift
        if self.width_shift_range > 0:
            tx = np.random.uniform(-self.width_shift_range, self.width_shift_range) * img_w
            x = shift(x, (0, tx, 0), order=1, mode='constant', cval=0.)

        # Zoom
        if self.zoom_range[0] != 1.0 or self.zoom_range[1] != 1.0:
            zx = zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zoomed_x = zoom(x, (zy, zx, 1), order=1)
            zh, zw = zoomed_x.shape[:2]

            if zy < 1.0:  # Zoom out - Pad
                h_pad = (img_h - zh) // 2
                w_pad = (img_w - zw) // 2
                padded_x = np.zeros_like(x)
                padded_x[h_pad:h_pad + zh, w_pad:w_pad + zw, :] = zoomed_x
                x = padded_x
            else:  # Zoom in - Crop
                h_crop = (zh - img_h) // 2
                w_crop = (zw - img_w) // 2
                x = zoomed_x[h_crop:h_crop + img_h, w_crop:w_crop + img_w, :]

        # Horizontal Flip
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = np.fliplr(x)

        # Vertical Flip
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = np.flipud(x)

        return x

    def flow(self, X, y, batch_size=32):
        """
        Generates batches of augmented data indefinitely.

        Args:
            X (np.ndarray): Input data (N, H, W, C).
            y (np.ndarray): Target data (N, ...).
            batch_size (int): Size of the batches to generate.

        Yields:
            tuple: A tuple (X_batch_augmented, y_batch).
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        while True:
            # Shuffle indices at the start of each epoch pass
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                X_batch_augmented = np.array([self._apply_random_transform(img) for img in X_batch])

                yield X_batch_augmented, y_batch
