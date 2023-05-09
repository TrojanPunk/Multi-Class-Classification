import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import ClassifierChain

df = pd.read_csv("emotions.csv")
print(df.head())

X = df.iloc[:, :-6]
y = df.iloc[:, -6:]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def build_model(model, mlb_estimator, xtrain, ytrain, xtest, ytest):
    # Create an Instance
    clf = mlb_estimator(model)
    clf.fit(xtrain, ytrain)
    # Predict
    clf_predictions = clf.predict(xtest)
    # Check For Accuracy
    acc = accuracy_score(ytest, clf_predictions)
    ham = hamming_loss(ytest, clf_predictions)
    result = {"accuracy:": acc, "hamming_score": ham}
    return result


clf_chain_model = build_model(SVC(kernel='linear', C=1.0), ClassifierChain, X_train, y_train, X_test, y_test)
print(clf_chain_model)
