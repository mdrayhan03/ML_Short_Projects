import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("email.csv")
x_data = data["Message"] # data for x parameter
y_data = data["Category"] # data for y parameter
# make the text data to vectorize data to train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x_data) # train and get new vocabulary
# spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y_data)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred))