import numpy as np
from sklearn.base import clone


class ClassifierChain:
    def __init__(self, base_estimator, order=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        self.estimators_ = []
        self.label_count = 0
        self.classes_ = None

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.order is None:
            self.order = np.arange(y.shape[1])
            np.random.shuffle(self.order)
        else:
            self.order = np.asarray(self.order)

        self.label_count = y.shape[1]
        self.estimators_ = [None] * self.label_count

        for label_index in self.order:
            # fit model for current label using all its predecessors
            X_augmented = self._augment_data(X, y, label_index)
            estimator = clone(self.base_estimator)
            estimator.fit(X_augmented, y[:, label_index])
            self.estimators_[label_index] = estimator

        self.classes_ = self.estimators_[-1].classes_

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.label_count))
        for label_index in self.order:
            X_augmented = self._augment_data(X, predictions, label_index)
            predicted_proba = self.estimators_[label_index].predict_proba(X_augmented)
            predictions[:, label_index] = predicted_proba[:, 1]

        return predictions

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], self.label_count))
        for label_index in self.order:
            X_augmented = self._augment_data(X, predictions, label_index)
            predicted_proba = self.estimators_[label_index].predict_proba(X_augmented)
            predictions[:, label_index] = predicted_proba[:, 1]

        return predictions

    def _augment_data(self, X, y, label_index):
        if label_index == 0:
            return X

        X_augmented = np.hstack([X, y[:, :label_index]])
        return X_augmented
