# Customer Churn Prediction

An end-to-end machine learning project to predict customer churn probability and identify the key factors driving customer attrition.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project develops a machine learning pipeline to predict whether a customer is likely to churn using demographic, service, account, and billing information.

The project covers the complete workflow:

* Exploratory Data Analysis
* Data preprocessing and feature engineering
* Machine learning model training
* Class imbalance handling
* Model evaluation and threshold optimization
* Model explainability using SHAP
* Interactive deployment using Streamlit

## Dataset

**Telco Customer Churn Dataset**

* 7,043 customer records
* 21 features
* Target variable: `Churn`

The dataset contains information about:

* Customer demographics
* Account and contract details
* Internet and subscribed services
* Payment methods
* Monthly and total charges
* Customer tenure

## Tech Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **XGBoost**
* **SHAP**
* **Matplotlib**
* **Streamlit**
* **Joblib**

## Machine Learning Workflow

### 1. Data Preprocessing

* Handled missing values
* Converted categorical variables using one-hot encoding
* Standardized numerical features
* Used `ColumnTransformer` to maintain a reproducible preprocessing pipeline

### 2. Model Training

The following models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

### 3. Model Evaluation

Models were evaluated using:

* ROC-AUC
* Precision
* Recall
* F1-score

Since identifying potential churners is important for customer retention, particular attention was given to the **precision–recall trade-off**.

### 4. Threshold Optimization

Instead of relying only on the default 0.5 classification threshold, different thresholds were evaluated to balance false positives and false negatives.

This allows businesses to adjust the model according to their retention strategy.

## Model Explainability

**SHAP (SHapley Additive Explanations)** was used to understand model predictions.

The analysis helps identify:

* Which features have the greatest influence on churn
* Why an individual customer receives a high churn probability
* Overall patterns associated with customer attrition

Key factors observed include:

* Contract type
* Customer tenure
* Monthly charges
* Internet service
* Additional support and security services

## Business Insights

The analysis suggests that:

* Customers on **month-to-month contracts** are more likely to churn.
* Customers with **shorter tenure** show higher churn risk.
* Higher **monthly charges** are associated with increased churn.
* Customers without services such as technical support and online security tend to show higher churn rates.

## Business Recommendations

Based on these findings, businesses could:

* Encourage customers to switch to longer-term contracts.
* Provide targeted retention offers to high-risk customers.
* Promote bundled support and security services.
* Focus retention efforts on newer customers.

## Streamlit Application

The trained model is deployed through a Streamlit application where users can enter customer information and receive a predicted churn probability.

The application also provides model insights to help interpret the prediction.

## Project Structure

```text
customer-churn-prediction-ml/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd customer-churn-prediction-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

## Project Outcome

This project demonstrates an end-to-end machine learning workflow combining **predictive modeling, model evaluation, explainability, and deployment** to solve a practical customer retention problem.
