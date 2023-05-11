import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


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

        for i, label_index in enumerate(self.order):
            # fit model for current label using all its predecessors
            estimator = LogisticRegression(solver='liblinear', random_state=self.random_state)
            if i == 0:
                estimator.fit(X, y[:, label_index])
            else:
                X_augmented = np.hstack([X, y[:, :i]])
                estimator.fit(X, y[:, [label_index]])
            self.estimators_[label_index] = estimator

        self.classes_ = y

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.label_count))
        for i, label_index in enumerate(self.order):
            if i == 0:
                predicted_proba = self.estimators_[label_index].predict_proba(X)
            else:
                X_augmented = np.hstack([X, predictions[:, :i]])
                predicted_proba = self.estimators_[label_index].predict_proba(X_augmented)
            predictions[:, label_index] = predicted_proba[:, 1]

        return predictions

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], self.label_count))
        for i, label_index in enumerate(self.order):
            if i == 0:
                predicted_proba = self.estimators_[label_index].predict_proba(X)
            else:
                X_augmented = np.hstack([X, predictions[:, :i]])
                predicted_proba = self.estimators_[label_index].predict_proba(X_augmented)
            predictions[:, label_index] = predicted_proba[:, 1]

        return predictions


# load the dataset
df = pd.read_csv("emotions.csv")

# split into features and labels
X = df.iloc[:, :-6]
y = df.iloc[:, -6:]

# create classifier chain
classifier_chain = ClassifierChain(LogisticRegression(solver='liblinear', random_state=42))

# fit the classifier chain to the data
classifier_chain.fit(X, y)

# predict on the training set
y_pred = classifier_chain.predict(X)

# calculate accuracy
accuracy = np.mean((y_pred == y).all(axis=1))

print("Accuracy:", accuracy)
