import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

class ClassifierChain(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_estimator, order=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Initialize the order of chains
        if self.order is None:
            self.order = np.arange(y.shape[1])
        else:
            self.order = np.array(self.order)

        # Fit the chains
        self.estimators_ = []
        self.estimators_.append(self.base_estimator)
        for i in range(1, y.shape[1]):
            y_subset = y[:, self.order[:i]]
            Xy = np.column_stack((X, y_subset))
            estimator = clone(self.base_estimator)
            self.estimators_.append(estimator.fit(Xy, y[:, self.order[i]]))
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Make predictions using the chain
        chain_predictions = np.zeros((X.shape[0], len(self.order)), dtype=np.int)
        for i, estimator in enumerate(self.estimators_):
            Xy = np.column_stack((X, chain_predictions[:, :i]))
            chain_predictions[:, i] = estimator.predict(Xy)

        # Return the final predictions
        return chain_predictions
