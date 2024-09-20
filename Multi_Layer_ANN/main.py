import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import Multi_Layer_ANN

if __name__ == "__main__":
    
    df = pd.read_csv("Dataset/Naive-Bayes-Classification-Data.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.ravel()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    hidden_layers = [5, 5]
    activations = ['relu', 'relu']
    
    ann = Multi_Layer_ANN(X_train, y_train, hidden_layers, activations, loss='binary_crossentropy')
    
    
    ann.fit(X_train, y_train, epochs=1000, learning_rate=0.05)
    y_pred = ann.predict(X_test)
    print(f"Predictions: {y_pred}")
    
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
