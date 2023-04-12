import numpy as np
from sklearn import datasets
# import matplotlib.pyplot as plt


class SVC:
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, max_iter: int = 1000) -> None:
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            for idx, xi in enumerate(X):
                gt_one = y_[idx] * (np.dot(xi, self.weight) - self.bias) >= 1
                if gt_one:
                    # update weight and bias
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight)
                else:
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight - np.dot(xi, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        output = np.dot(X, self.weight) - self.bias
        return np.sign(output)  # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0


if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)
    y = np.where(y == 0, -1, 1)

    svc_clf = SVC()
    svc_clf.fit(X, y)
    y_pred = svc_clf.predict(X)
    accuracy = np.mean(y == y_pred)

    print(f'Accuracy score: {accuracy * 100:.2f}%')
    print(svc_clf.weight, svc_clf.bias)

### Vizualise the predictions with hiperplanes ###

# def visualize_svm():
#     def get_hyperplane_value(x, w, b, offset):
#         return (-w[0] * x + b + offset) / w[1]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
#
#     x0_1 = np.amin(X[:, 0])
#     x0_2 = np.amax(X[:, 0])
#
#     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
#     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)
#
#     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
#     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
#
#     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
#     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)
#
#     ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
#     ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
#     ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")
#
#     x1_min = np.amin(X[:, 1])
#     x1_max = np.amax(X[:, 1])
#     ax.set_ylim([x1_min - 3, x1_max + 3])
#
#     plt.show()
#
#
# visualize_svm()
