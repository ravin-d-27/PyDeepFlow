import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_training_history(history, metrics=('loss','accuracy'), figure='test_training_history.png'):
    num_metrics = len(metrics)
    fig = Figure(figsize=(6 * num_metrics, 5))
    canvas = FigureCanvas(fig)

    if num_metrics == 1:
        ax = [fig.add_subplot(1, 1, 1)]
    else:
        ax = [fig.add_subplot(1, num_metrics, i + 1) for i in range(num_metrics)]

    for i, metric in enumerate(metrics):
        ax[i].plot(history.get(f'train_{metric}', []), label=f'Train {metric.capitalize()}')
        if f'val_{metric}' in history:
            ax[i].plot(history.get(f'val_{metric}', []), label=f'Validation {metric.capitalize()}')
        ax[i].set_title(f"{metric.capitalize()} over Epochs")
        ax[i].set_xlabel("Epochs")
        ax[i].set_ylabel(metric.capitalize())
        ax[i].legend()

    fig.tight_layout()
    fig.savefig(figure)


if __name__ == '__main__':
    hist = {
        'train_loss': [1, 0.5, 0.3],
        'val_loss': [1.1, 0.6, 0.4],
        'train_accuracy': [0.1, 0.5, 0.9],
        'val_accuracy': [0.05, 0.55, 0.88]
    }
    plot_training_history(hist, metrics=('loss','accuracy'), figure='test_training_history.png')
    import os
    print('SAVED' if os.path.exists('test_training_history.png') else 'FAILED')
