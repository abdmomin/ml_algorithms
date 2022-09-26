import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##### Formula #####

## P(y|X) = P(X|y) . P(y) / P(X)
## -> P(y|X) = P(x1|y) . P(x2|y)...P(xi|y) . P(y) / P(X)
## -> P(y|X) = log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xi|y)) + log(P(y))
## -> P(y|X) = argmax(log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xi|y)) + log(P(y)))

## -> P(y) = len(class) / sample_size [freaquency of the class]

## -> P(xi|y) = 1 / sqrt(2*pi*variance) * exp(-(xi - mean)**2 / 2*variance) [PDF]

class NaiveBayesClassifier(object):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self.py = np.zeros(n_classes, dtype=np.float64) ## --> P(y) / Prior

        for cls in self.classes:
            X_cls = X[y==cls]
            self.mean[cls, :] = X_cls.mean(axis=0)
            self.variance[cls, :] = X_cls.var(axis=0)
            self.py[cls] = X_cls.shape[0] / float(n_samples)

    def predict(self, X):
        predictions = [self._probability(x) for x in X]
        return predictions

    def _probability(self, x):
        pyx = [] ## --> P(y|X) / Posterior
        
        for idx, cls in enumerate(self.classes):
            py = np.log(self.py[idx]) ## --> P(y)
            pxy = np.sum(np.log(self._pdf(idx, x))) ## --> P(X|y) / Class Conditionals
            posterior = pxy + py ## --> sum(log(P(X|y))) + log(P(y))
            pyx.append(posterior)
        return self.classes[np.argmax(pyx)]

    def _pdf(self, idx, x):
        mu = self.mean[idx]
        var = self.variance[idx]

        numerator = np.exp(-(x - mu)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator


if __name__ == '__main__':
    nbclf = NaiveBayesClassifier()
    nbclf.fit(X_train, y_train)
    preds = nbclf.predict(X_test)

    accuracy = np.sum(preds == y_test) / len(y_test)

    print(f'Accuracy: {accuracy*100:.2f}%')