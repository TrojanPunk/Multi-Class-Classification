import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import hamming_loss, accuracy_score
from ClassifierChain import ClassifierChain

# Load the dataset
data = pd.read_csv('emotions.csv')

# Extract the features and labels
X = data.iloc[:, :-6].values
y = data.iloc[:, -6:].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Binarize the labels
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)

# Create an instance of GradientBoostingClassifier
gbc = GradientBoostingClassifier()

# Fit the classifier on the training data
gbc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gbc.predict(X_test)

# Evaluate the performance of the classifier
hl = hamming_loss(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f'Hamming Loss: {hl:.4f}')
print(f'Accuracy Score: {acc:.4f}')

# Create an instance of ClassifierChain
cc = ClassifierChain(base_estimator=gbc)

# Fit the classifier on the training data
cc.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_cc = cc.predict(X_test)

# Evaluate the performance of the classifier chain
hl_cc = hamming_loss(y_test, y_pred_cc)
acc_cc = accuracy_score(y_test, y_pred_cc)
print(f'Hamming Loss (Classifier Chain): {hl_cc:.4f}')
print(f'Accuracy Score (Classifier Chain): {acc_cc:.4f}')
