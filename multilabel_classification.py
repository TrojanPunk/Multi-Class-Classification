import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import ClassifierChain
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, hamming_loss

# Load the emotions dataset
data = pd.read_csv('emotions.csv')

# Split the data into features and labels
X = data.iloc[:, :-6].values  # features
y = data.iloc[:, -6:].values  # labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

# Train the classifier chain model using the neural network as the base classifier
classifier = ClassifierChain(model)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the model using accuracy and hamming loss metrics
accuracy = accuracy_score(y_test, y_pred)
hamming_loss = hamming_loss(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Hamming Loss: {hamming_loss}")
