import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MyClassifierChain:
    
    def __init__(self, base_estimator, order=None):
        self.base_estimator = base_estimator
        self.order = order
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        
        self.estimators_ = []
        self.order_ = self.order
    
        if self.order_ is None:
            self.order_ = np.arange(n_outputs)
    
        X_augmented = X
    
        for i in self.order_:
            estimator = clone(self.base_estimator)
            y_subset = y[:, :i+1]
        
            estimator.fit(X_augmented, y_subset)
            self.estimators_.append(estimator)
            X_augmented = np.hstack((X_augmented, y_subset))
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n_samples, n_features = X.shape
        n_outputs = len(self.estimators_)
        y_pred = np.zeros((n_samples, n_outputs), dtype=np.int)
        X_augmented = X
        
        for i, estimator in enumerate(self.estimators_):
            y_pred_subset = estimator.predict(X_augmented)
            y_pred[:, i] = y_pred_subset.ravel()
            X_augmented = np.hstack((X_augmented, y_pred_subset))
        
        return y_pred
