import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

class ClassifierChain(BaseEstimator, ClassifierMixin):
    
    def __init__(self, base_estimator, order=None, random_state=None):
        """
        This is the constructor method for the ClassifierChain class. It takes three arguments: base_estimator, which is the estimator used as the base model in the chain; 
        order, which is an optional argument specifying the order in which the chains are applied; and random_state, which is an optional argument for setting the random seed. 
        The method initializes these attributes as instance variables.
        """

        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state

    def fit(self, X, y):
        """
        This method fits the classifier on the training data. It takes two arguments: X, which is the feature matrix, and y, which is the target matrix. 
        The method first checks that the input has the correct shape and stores the unique labels in the classes_ attribute. It then initializes the order of the chains if 
        it was not specified in the constructor, and fits each chain using the base estimator. The method returns the fitted estimator.
        """
        # Check that X and y have correct shape
        y = y.ravel()
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
        """
        This method generates predictions on new data using the fitted estimator. It takes one argument: X, which is the feature matrix. 
        The method checks if fit has been called, and then generates predictions using the chains. It first initializes an empty array to hold the chain predictions, 
        and then iterates over the estimators to generate the chain predictions. The final predictions are returned.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Make predictions using the chain
        chain_predictions = np.zeros((X.shape[0], len(self.order)), dtype=np.int)
        for i, estimator in enumerate(self.estimators_):
            Xy = np.column_stack((X, chain_predictions[:, :i]))
            chain_predictions[:, i] = estimator.predict(Xy)

        # Return the final predictions
        return chain_predictions
