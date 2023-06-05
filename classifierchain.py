import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import LabelBinarizer

class ClassifierChain(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifiers = []

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=True)

        check_classification_targets(y)
        self.label_binarizer_ = LabelBinarizer()
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

        for i in range(Y.shape[1]):
            classifier = self.base_classifier
            y_subset = np.concatenate((X, Y[:, :i]), axis=1)
            if sample_weight is not None:
                sample_weight = compute_sample_weight('balanced', y_subset, sample_weight=sample_weight)
                classifier.fit(X, Y[:, i], sample_weight=sample_weight)
            else:
                classifier.fit(X, Y[:, i])
            self.classifiers.append(classifier)
            X = np.concatenate((X, Y[:, :i]), axis=1)

        return self

    def predict(self, X):
        check_consistent_length(X)
        X = check_array(X, accept_sparse=True)

        Y = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        for i, classifier in enumerate(self.classifiers):
            Y[:, i] = classifier.predict(X)
            X = np.concatenate((X, Y[:, :i+1]), axis=1)

        return self.label_binarizer_.inverse_transform(Y)

    def predict_proba(self, X):
        check_consistent_length(X)
        X = check_array(X, accept_sparse=True)

        Y = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for i, classifier in enumerate(self.classifiers):
            Y[:, i] = classifier.predict_proba(X)[:, 1]
            X = np.concatenate((X, Y[:, :i+1]), axis=1)

        return Y
