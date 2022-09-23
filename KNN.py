from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class KNNeighbors(object):
    def __init__(self, K: int) -> None:
        self.K = K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predictions = [self._knn(x) for x in X]
        return np.array(predictions)

    def _knn(self, x):
        distences = [self._euclidean_distance(
            x, x_train) for x_train in self.X_train]

        indices = np.argsort(distences)[:self.K]
        labels = [self.y_train[i] for i in indices]

        nearest_neighbors = Counter(labels).most_common(1)[0][0]

        return nearest_neighbors

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # plt.figure(figsize=(10, 7))
    # plt.scatter(X[:, 0], X[:, 1], c=y_train, edgecolor='k')
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    knn = KNNeighbors(K=3)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)

    print(f"Accuracy: {accuracy*100:.2f}%")
