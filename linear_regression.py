from turtle import clear
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
    n_samples=200, n_features=1, n_targets=1, noise=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X_train.shape, y_train.shape)
# plt.scatter(X_train, y_train, c='b', s=20)
# plt.show()


class LinearRegression(object):
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_hat = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        return predictions


if __name__ == '__main__':
    linear = LinearRegression(lr=0.01)
    linear.fit(X_train, y_train)

    y_pred = linear.predict(X_test)

    mse = np.mean((y_pred - y_test)**2)
    print(f'MSE: {mse}')

    all_preds = linear.predict(X)
    mse = np.mean((all_preds - y)**2)

    plt.scatter(X_train, y_train, c='b', s=20)
    plt.scatter(X_test, y_test, c='b', s=20)
    plt.plot(X, all_preds, c='r', linewidth=2, label='Predictions')
    plt.title(f'MSE Loss: {mse:.3f}', fontsize=15)
    plt.show()
