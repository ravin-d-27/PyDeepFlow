import pandas as pd
import numpy as np

class NaiveBayesClassifier:

    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = None
        self.feature_stats = {}

    def calculate_prior(self, y):
        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        for c in class_counts:
            self.priors[c] = class_counts[c] / total_samples


    def calculate_likelihoods(self, X, y):
        self.classes = y.unique()
        self.feature_stats = {c: {} for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            for feature in X.columns:
                
                feature_mean = X_c[feature].mean()
                feature_var = X_c[feature].var()
                self.feature_stats[c][feature] = (feature_mean, feature_var)

    
    def gaussian_likelihood(self, x, mean, var):
        eps = 1e-4  
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-(np.power(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent


    def fit(self, X, y):
        self.calculate_prior(y)
        self.calculate_likelihoods(X, y)

    
    def predict_single(self, x):
        posteriors = {}
        for c in self.classes:
            posterior = np.log(self.priors[c])  # log to avoid underflow
            for feature in x.index:
                mean, var = self.feature_stats[c][feature]
                posterior += np.log(self.gaussian_likelihood(x[feature], mean, var))
            posteriors[c] = posterior

        return max(posteriors, key=posteriors.get)

    
    def predict(self, X):
        predictions = X.apply(self.predict_single, axis=1)
        return predictions
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred) * 100



def load_data(csv_file):
    data = pd.read_csv(csv_file)
    print(data.head())
    return data

def train_and_evaluate(csv_file):
    
    data = load_data(csv_file)
    
    X = data.iloc[:, :-1] 
    y = data.iloc[:, -1]  
    train_size = int(0.8 * len(data))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    classifier = NaiveBayesClassifier()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = classifier.accuracy(y_test, y_pred)
    print("The Accuracy is : ",accuracy,"%")


csv_file = 'E:/Github_Repos/Artificial_Neural_Networks_From_Scratch/Naive_Bayes/Naive-Bayes-Classification-Data.csv' 
train_and_evaluate(csv_file)
