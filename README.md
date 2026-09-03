# Live Demo 🚀
Coba langsung aplikasinya di sini:

- **App (Logistic Regression):** [customer-churn-prediction-gqjnxwqubafvp7lwhyf9s.streamlit.app](https://customer-churn-prediction-gqjnxvwqubafvp7lwhyf9s.streamlit.app)
### Fitur
- 📊 Overview & KPI dashboard customer churn
- 📈 Churn analytics berdasarkan contract, payment method, tenure
- 🔮 Prediksi churn customer secara real-time berdasarkan input data

# 📊 Customer Churn Prediction

A Machine Learning project to predict customer churn using **Logistic Regression**, with a comparison between a baseline model and a model trained using **SMOTE** to handle class imbalance.

The project also includes an interactive web application built with **Streamlit** for customer churn prediction.

---

## 🚀 Project Overview

Customer churn is an important problem for businesses because losing existing customers can directly affect revenue and long-term growth.

This project aims to build a classification model that predicts whether a customer is likely to churn based on their demographic, service, contract, and billing information.

Two Logistic Regression approaches were developed and compared:

1. **Logistic Regression Baseline**
2. **Logistic Regression + SMOTE**

The goal is not only to maximize accuracy, but also to improve the model's ability to identify customers who are likely to churn.

---

## 🎯 Objectives

- Analyze customer churn patterns.
- Perform exploratory data analysis (EDA).
- Prepare and preprocess the dataset.
- Build a Logistic Regression baseline model.
- Handle class imbalance using SMOTE.
- Compare model performance.
- Evaluate models using multiple classification metrics.
- Deploy the prediction model using Streamlit.

---


### 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── app_smote.py
│
├── churn.ipynb
│
├── churn_model.pkl
├── churn_smote_model.pkl
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

### 🎯 Business Problem

Customer churn is an important business problem for telecommunication companies.

Identifying customers who are likely to churn can help companies develop targeted retention strategies and reduce customer loss.

**This project aims to build a machine learning classification model that predicts whether a customer is likely to churn based on their demographic, service, contract, and billing information.**

---
