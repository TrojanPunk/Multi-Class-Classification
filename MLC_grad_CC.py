import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from CC import ClassifierChain

# Load the emotions dataset
emotions_df = pd.read_csv('emotions.csv')

# Extract the features and labels
X = emotions_df.iloc[:, :-6].values
y = emotions_df.iloc[:, -6:].values

# Binarize the labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the MultiOutputClassifier model
base_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
classifier_chain = ClassifierChain(base_model, order=[0, 1, 2, 3, 4, 5], random_state=42)

# Fit the model on the training data
classifier_chain.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier_chain.predict(X_test)

# Evaluate the model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
