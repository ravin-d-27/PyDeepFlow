class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience:int = 5, delta:float = 0) -> None:
        """
        Initializes the EarlyStopping.
        
        :param patience (int): How long to wait after last time validation loss improved.
                               Default: 5
        :param delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                              Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_score = None
        self.counter = 0
    def __call__(self, val_loss:float) -> None:
        """
        Checks if the condition for early stopping are met and updates early_stop flag
        
        :param val_loss (float): The current validation loss.
        """
        score = - val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter+=1
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
