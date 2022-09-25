import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = datasets.load_breast_cancer()

X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#### Formula #####
## linear = W * X + b
## sigmoid = 1 / (1 + exp(-linear))

## Gradient descent ##
## dw = 1/N * sum(2 * X * (y_hat - y))
## db = 1/N * sum(2 * (y_hat - y))

## Update weight and bias ##
## weight = weight - liarning_rate * dw
## bias = bias - liarning_rate * db

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            h = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(h)

            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        h = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(h)
        predictions = [1 if i > 0.5 else 0 for i in y_hat]
        return np.array(predictions)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    logreg = LogisticRegression(lr=0.001)
    logreg.fit(X_train, y_train)
    preds = logreg.predict(X_test)

    accuracy = np.sum(preds == y_test)/len(y_test)

    print(f'Accuracy: {accuracy*100:.2f}%')
