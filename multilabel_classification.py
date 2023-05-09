# Import required libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer

# Load the data
data = pd.read_csv('emotions.csv')

# Split data into features and labels
X = data.iloc[:,:-6]
y = data.iloc[:,-6:]

# Convert labels to binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y.values)

# Define the model
def create_model():
    model = Sequential()
    model.add(Dense(6, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the classifier
classifier = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10)

# Evaluate the classifier using k-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(classifier, X, y, cv=kfold)

# Print the results
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
