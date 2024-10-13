class EarlyStopping:
    def __init__(self, patience:int = 5, delta:int = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_score = None
        self.counter = 0
    def __call__(self, val_loss:float) -> None:
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
