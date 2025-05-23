import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("email.csv")
x_data = data["Message"] # data for x parameter
y_data = data["Category"] # data for y parameter
# make the text data to vectorize data to train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x_data) # train and get new vocabulary
# spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y_data)

svm_model = SVC(kernel='linear')  # linear kernel works well for text
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("SVM Classification Report:\n", classification_report(y_test, y_pred))