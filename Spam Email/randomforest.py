import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("email.csv")
X_text = data["Message"]
y = data["Category"]

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameters grid to search
param_grid = {
    'n_estimators': [50, 100, 200],        # number of trees
    'max_depth': [None, 10, 20, 30],      # max depth of each tree
    'min_samples_split': [2, 5, 10],      # minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],        # minimum samples required at leaf node
    'bootstrap': [True, False]             # whether bootstrap samples are used
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,               # 3-fold cross-validation
    n_jobs=-1,          # use all processors
    verbose=2,
    scoring='accuracy'  # metric to optimize
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Evaluate the best estimator on test data
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))