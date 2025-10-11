class EarlyStopping:
    """Early stops the training if the monitored metric doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, delta: float = 0, mode: str = 'min') -> None:
        """
        Initializes the EarlyStopping.
        
        :param patience (int): How long to wait after last time validation metric improved.
                               Default: 5
        :param delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                              Default: 0
        :param mode (str): One of {'min', 'max'}. In 'min' mode (Loss), training stops
                           when metric stops decreasing. In 'max' mode (Accuracy),
                           training stops when metric stops increasing.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode.lower()
        self.early_stop = False
        self.counter = 0

        if self.mode == 'min':
            self.best_score = float('inf') 
        elif self.mode == 'max':
            self.best_score = -float('inf') 
        else:
            self.best_score = float('inf')
            self.mode = 'min'

    def __call__(self, current_metric: float) -> None:
        """
        Checks if the condition for early stopping are met and updates early_stop flag
        
        :param current_metric (float): The current metric (e.g., val_loss, val_accuracy).
        """
        
        if not (isinstance(current_metric, (int, float)) and float('-inf') < current_metric < float('inf')):
             self.early_stop = True
             return
        
        is_better = False
        
        if self.mode == 'min':
            is_better = current_metric < self.best_score - self.delta
        elif self.mode == 'max':
            is_better = current_metric > self.best_score + self.delta
        
        if is_better:
            self.best_score = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
