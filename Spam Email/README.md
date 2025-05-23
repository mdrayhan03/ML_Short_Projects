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