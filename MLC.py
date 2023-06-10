# Program that create a classifier chain and perform multilabel classification.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels


class ClassifierChain:
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifiers = []

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = unique_labels(y)

        for i in range(y.shape[1]):
            classifier = self.base_classifier
            classifier.fit(X, y[:, i])
            self.classifiers.append(classifier)

            # Augment feature matrix
            X = np.concatenate((X, y[:, :i]), axis=1)

    def predict(self, X):
        X = check_array(X, accept_sparse=True)

        Y_pred = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        for i, classifier in enumerate(self.classifiers):
            Y_pred[:, i] = classifier.predict(X)
            X = np.concatenate((X, Y_pred[:, :i+1]), axis=1)

        return Y_pred

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=True)

        Y_pred_proba = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for i, classifier in enumerate(self.classifiers):
            Y_pred_proba[:, i] = classifier.predict_proba(X)[:, 1]
            X = np.concatenate((X, Y_pred_proba[:, :i+1]), axis=1)

        return Y_pred_proba


# Load the emotions dataset from CSV
emotions_data = pd.read_csv('emotions.csv')

# Extract features (X) and labels (y)
X = emotions_data.iloc[:, :-6].values
y = emotions_data.iloc[:, -6:].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base classifier
base_classifier = RandomForestClassifier()

# Build the classifier chain
classifier_chain = ClassifierChain(base_classifier)

# Train the classifier chain
classifier_chain.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier_chain.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
