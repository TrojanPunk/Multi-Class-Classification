import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Classifier_chain import *

# Downloading and loading the Yeast dataset
yeast_df = pd.read_csv('emotions.csv')

# Splitting into X and y
X = yeast_df.iloc[:, :-6]
y = yeast_df.iloc[:, -6:]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.values.ravel()

# Creating the custom classifier chain with RandomForestClassifier as base estimator
classifier = MyClassifierChain(RandomForestClassifier(random_state=42))

# Fitting the classifier to the training data
classifier.fit(X_train, y_train)

# Predicting the test set
y_pred = classifier.predict(X_test)

# Printing the accuracy score
print('Accuracy score:', classifier.score(X_test, y_test))
