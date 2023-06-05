import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import ClassifierChain

# Load the emotions dataset from CSV
emotions_data = pd.read_csv('emotions.csv')

# Split the dataset into features (X) and labels (y)
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
