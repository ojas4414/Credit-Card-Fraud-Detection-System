# 💳 Credit Card Fraud Detection System

## Overview

This project implements a machine learning–based fraud detection system that predicts whether a credit card transaction is fraudulent.

It combines a trained **XGBoost model**, exploratory analysis notebooks, and an interactive **Streamlit web app** for real-time fraud risk prediction.

---

## 🚀 Features

* Fraud probability prediction using **XGBoost**
* Interactive dashboard built with **Streamlit**
* Quick demo with real fraud & normal transactions
* Real-time risk classification
* Clean fintech-style UI
* End-to-end ML pipeline (data → training → deployment)

---

## 📁 Project Structure

```
CREDIT-CARD-FRAUD-ML/
│
├── app.py                         # Streamlit web app
│
├── models/
│   └── xgboost_fraud_model.pkl   # Trained ML model
│
├── notebooks/
│   ├── 02_DecisionTreeClassifier.ipynb   # model training & analysis
│   └── data/
│       └── creditcard.csv        # transaction dataset
│
├── requirements.txt              # dependencies
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/ojas4414/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System

```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## ⚡ Using the App

### 🟢 Load Normal Transaction

Loads a genuine transaction example.

### 🔴 Load Fraud Transaction

Loads a fraudulent transaction example.

### 💰 Predict Fraud Risk

Displays the probability that the transaction is fraudulent.

---

## 📊 Features Used

The model evaluates transactions using:

* **Time**
* **Amount**
* **V1–V28 anonymized features**

These features are PCA-transformed variables to protect user privacy while preserving transaction patterns.

---

## 🧠 Model

* Algorithm: **XGBoost Classifier**
* Task: Binary classification (Fraud vs Genuine)
* Output: Fraud probability score

---

## 📚 Dataset

This project uses the widely studied European cardholders dataset for fraud detection research.

Due to privacy protection, original feature meanings are anonymized.

---

## 🌍 Why This Project Matters

Credit card fraud detection is a critical real-world application of machine learning.
This project demonstrates how predictive models can be integrated into real-time systems to support secure digital transactions.

---

## 🔮 Future Improvements

* Explainable AI for fraud reasoning
* Real-time API deployment
* Model retraining pipeline
* Dashboard analytics & trends
* Cloud deployment

---

## 👨‍💻 Author

Built to explore applied machine learning, fintech security, and real-time fraud detection systems.
