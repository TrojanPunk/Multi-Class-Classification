from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from ClassifierChain import ClassifierChain

# Generate a random multi-label classification problem
X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=5, n_labels=3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the classifier chain
clf_chain = ClassifierChain(LogisticRegression())
clf_chain.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_chain.predict(X_test)

# Calculate the hamming loss
hamming_loss = hamming_loss(y_test, y_pred)
print("Hamming Loss:", hamming_loss)

# Transform the true and predicted labels into binary format
mlb = MultiLabelBinarizer()
y_test_bin = mlb.fit_transform(y_test)
y_pred_bin = mlb.transform(y_pred)

# Calculate the accuracy score
accuracy_score = mlb.score(y_test_bin, y_pred_bin)
print("Accuracy Score:", accuracy_score)
