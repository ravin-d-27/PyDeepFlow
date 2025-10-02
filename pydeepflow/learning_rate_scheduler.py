import numpy as np


class LearningRateScheduler:
    def __init__(
        self, initial_lr, strategy="decay", decay_rate=0.1, cycle_length=10, min_lr=1e-6
    ):
        """
        Initializes the LearningRateScheduler.

        :param initial_lr: Initial learning rate.
        :param strategy: The strategy to use, either 'decay' or 'cyclic'.
        :param decay_rate: Decay rate for exponential decay.
        :param cycle_length: Number of epochs for one learning rate cycle (used for cyclic strategy).
        :param min_lr: Minimum learning rate value to avoid too low values.
        """
        self.initial_lr = initial_lr
        self.strategy = strategy
        self.decay_rate = decay_rate
        self.cycle_length = cycle_length
        self.min_lr = min_lr
        self.current_lr = initial_lr

    def get_lr(self, epoch):
        """
        Returns the learning rate for the current epoch based on the selected strategy.

        :param epoch: The current epoch number.
        :return: The learning rate for the current epoch.
        """
        if self.strategy == "decay":
            # Exponential decay: LR = initial_lr * (decay_rate ^ epoch)
            lr = self.initial_lr * (self.decay_rate**epoch)
            return max(lr, self.min_lr)  # Ensure the LR doesn't fall below min_lr
        elif self.strategy == "cyclic":
            # Cyclic learning rate
            cycle_position = epoch % self.cycle_length
            lr = (
                self.min_lr
                + (self.initial_lr - self.min_lr)
                * (1 + np.cos(np.pi * cycle_position / self.cycle_length))
                / 2
            )
            return lr
        else:
            raise ValueError(
                "Invalid learning rate strategy. Choose 'decay' or 'cyclic'."
            )
