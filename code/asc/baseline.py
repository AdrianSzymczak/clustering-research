import numpy as np

class AllInOneBaseline:
    def fit(self, X):
        yield

    def transform(self, X):
        return np.ones(X.shape[0])

    def fit_transform(self, X):
        return self.transform(X)

class SingletonBaseline:

    def fit(self, X):
        yield

    def transform(self, X):
        return np.arange(X.shape[0]) * 1.0

    def fit_transform(self, X):
        return self.transform(X)