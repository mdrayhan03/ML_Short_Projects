# 📧 Email Spam Classifier

This project is a machine learning-based **Email Spam Classifier** built using **Logistic Regression** and **TF-IDF Vectorization**. It determines whether a given email message is **spam** or **ham** (not spam).

---

## 🚀 Features

- Classifies emails as `spam` or `not spam`
- Trained using **Logistic Regression**
- Utilizes **TF-IDF** to vectorize raw text
- 97%+ accuracy on test data
- Saves and loads model using `joblib` for real-world predictions

---

## 🧠 Technologies Used

- Python
- pandas
- scikit-learn
- joblib

---

## 🗂️ Dataset Format

The model is trained on a CSV file named `email.csv` with the following columns:

| Message                            | Category |
|------------------------------------|----------|
| "Free entry in a prize draw"       | spam     |
| "Let's catch up tomorrow"          | ham      |

---

## 📦 Installation

Make sure you have Python installed. Then install dependencies:

```bash
pip install scikit-learn pandas joblib
```
## 🛠️ How to Run

1. Place your dataset in the same directory as `email.csv`.

2. Run the script:

```bash
python spam_classifier.py
```
3. Input any email content when prompted:
```text
Enter your mail: Congratulations, you’ve won a free ticket!
Your email is spam
```
## 📈 Accuracy

Achieved an accuracy score of:

```text
Accuracy Score: 97.34%
```

### 8. Saved Files
## 💾 Saved Files

- `spam_logistic_model.pkl`: Trained model  
- `vectorize_x_data.pkl`: TF-IDF vectorizer  

You can use these files later to make predictions without retraining.
## 📬 Sample Prediction Function

```python
def real_predict(text):
    vect_text = vector.transform([text])
    predict = load_model.predict(vect_text)
    print(f"Your email is {'not spam' if predict[0] == 'ham' else 'spam'}")
```
