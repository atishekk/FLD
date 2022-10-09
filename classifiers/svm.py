from sklearn import svm

class LinearSVM:
    def __init__(self, **kwargs) -> None:
        self.model = svm.LinearSVC(kwargs)

    def fit(self, X, y):
        return self.model.fit(X, y)

