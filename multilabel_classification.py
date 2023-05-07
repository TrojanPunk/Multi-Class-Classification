from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# generate sample dataset
X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=5, random_state=42)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create classifier chains object with base estimator as decision tree
classifier = ClassifierChain(base_estimator=DecisionTreeClassifier(random_state=42), order='random', random_state=42)

# train the model
classifier.fit(X_train, y_train)

# make predictions on test set
y_pred = classifier.predict(X_test)

# evaluate performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

print("Accuracy:", accuracy)
print("F1 score:", f1)
