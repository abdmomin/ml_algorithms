import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


X, y = datasets.make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class Perceptron(object):
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        y = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.iters):
            for idx, x in enumerate(X):
                linear = np.dot(x, self.weights) + self.bias
                output = y[idx] - self.unit_step_fn(linear)
                self.weights += self.lr * output * x
                self.bias += self.lr * output

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.unit_step_fn(linear)

    def unit_step_fn(self, x):
        return np.where(x >= 0, 1, 0)



if __name__ == '__main__':
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_hat = perceptron.predict(X_test)

    accuracy = np.sum(y_hat == y_test) / len(y_test)

    print(f'Accuracy: {accuracy*100:.2f}%')