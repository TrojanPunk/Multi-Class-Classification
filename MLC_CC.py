import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from Classifier_chain import ClassifierChain
from sklearn.svm import SVC

# Load the data
data = pd.read_csv('emotions.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-6],
                                                    data.iloc[:, -6:],
                                                    test_size=0.2,
                                                    random_state=42)

# Train the classifier chains model
classifier = ClassifierChain(SVC(kernel='linear', probability=True, random_state=42), order='random', random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the model
score = classifier.score(X_test, y_test)

print(f"Classifier chains accuracy: {score}")
