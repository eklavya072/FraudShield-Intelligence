# 🛡️ FraudShield Intelligence

**Enterprise-Grade Real-Time Financial Fraud Detection Platform**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1798e6?style=flat&logo=xgboost)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

FraudShield Intelligence is an end-to-end machine learning platform engineered to detect fraudulent financial transactions with ultra-low latency. By pairing an optimized extreme gradient boosting framework with a scalable asynchronous REST API and an intuitive analytical dashboard, the platform delivers real-time risk scoring alongside model explainability powered by Explainable AI (XAI) paradigms.

---

##  Live Demo

| Service | URL |
|---|---|
| **Application** | https://fraudshield-intelligence-evci2hh3g7wzrytqmg848y.streamlit.app |


---

##  Overview

FraudShield Intelligence is a production-ready machine learning system designed to identify potentially fraudulent financial transactions in real time. The project goes beyond a single model — it systematically trains and benchmarks three classifiers (Logistic Regression, Random Forest, and XGBoost) on a 6.3M+ transaction dataset, with XGBoost selected as the final model based on superior recall and F1 performance on the highly imbalanced fraud detection task.

The platform combines a high-performance classification backend with a modern web interface and REST API, enabling users to evaluate transaction risk through an intuitive dashboard — with SHAP-powered explanations for every prediction.

---

## Application Preview

### Home Page


<img width="1708" height="1017" alt="Screenshot 2026-06-29 at 2 49 45 AM" src="https://github.com/user-attachments/assets/074bcbfe-bc18-4a4a-834a-e56afca6f262" />


### Prediction Result

<img width="1709" height="1014" alt="Screenshot 2026-06-29 at 2 51 08 AM" src="https://github.com/user-attachments/assets/fb83b3f0-26b2-428a-b0f2-98038d0fea22" />


### SHAP Explainability
<img width="680" height="814" alt="Screenshot 2026-06-29 at 2 53 18 AM" src="https://github.com/user-attachments/assets/1562f901-f5af-436e-9cd3-b8260fae4c1f" />


---

##  Features

- Real-time fraud prediction via REST API
- Rigorous multi-model training and evaluation pipeline
- XGBoost classifier selected through benchmarked comparison
- Interactive Streamlit frontend dashboard
- FastAPI backend with Swagger UI documentation
- SHAP Explainable AI — per-prediction feature attribution
- Docker and Docker Compose support
- Cloud deployment (Render + Streamlit Community Cloud)
- Confidence score visualization
- Transaction summary dashboard

---

##  Model Selection & Empirical analysis

Three classifiers were trained and rigorously evaluated to identify the best performer for fraud detection — where **recall** is the critical metric due to the severe class imbalance inherent in financial fraud datasets.

### Model Comparison

| Metric | Logistic Regression | Random Forest | **XGBoost** |
|---|---|---|---|
| Accuracy | Baseline | 99.97% | **99.98%** |
| Precision | Baseline | 98.02% | 96.60% |
| Recall | Baseline | 78.59% | **85.98%** |
| F1 Score | Baseline | 87.24% | **90.98%** |
| False Negatives | High | 446 | **292** |

### XGBoost — Selected Model

| Metric | Score |
|--------|-------|
| Accuracy | 99.978% |
| Precision | 96.60% |
| **Recall** | **85.98%** |
| **F1 Score** | **90.98%** |

```
Confusion Matrix:
                   Predicted Legitimate   Predicted Fraud
Actual Legitimate        1,588,509               63
Actual Fraud                   292            1,791
```

XGBoost was selected for its superior recall and F1 score — catching the highest proportion of actual fraudulent transactions while maintaining strong precision. Its gradient boosting framework handles class imbalance more effectively, delivering a **7.4-point recall improvement** over Random Forest and catching **154 additional fraudulent transactions** per evaluation cycle.

---

##  System Architecture

```
           User
             │
             ▼
  Streamlit Frontend
  (Streamlit Community Cloud)
             │
       REST API Call
             │
             ▼
    FastAPI Backend
       (Render)
             │
             ▼
  XGBoost Fraud Detection Model
       + SHAP Explainer
             │
             ▼
  Fraud Prediction + Feature Attributions
```

---

##  Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Backend | FastAPI |
| Frontend | Streamlit |
| Explainability | SHAP |
| API Documentation | Swagger UI |
| Containerization | Docker & Docker Compose |
| Cloud Deployment | Render (backend), Streamlit Community Cloud (frontend) |
| Version Control | Git & GitHub |

---

##  Input Features

| Feature | Description |
|---|---|
| Step | Hours elapsed since first transaction |
| Transaction Type | CASH_IN, CASH_OUT, PAYMENT, DEBIT, TRANSFER |
| Amount | Transaction amount (USD) |
| Sender Balance Before | Sender's account balance before transaction |
| Sender Balance After | Sender's account balance after transaction |
| Receiver Balance Before | Receiver's account balance before transaction |
| Receiver Balance After | Receiver's account balance after transaction |
| Flagged Fraud | Rule-based fraud indicator from source system |

---

##  Project Structure

```
FraudShield-Intelligence/
│
├── backend/
│   ├── app.py                          # FastAPI application & prediction endpoints
│   ├── credit_fraud_xgb.pkl            # Trained XGBoost model artifact
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── streamlit_app.py                # Streamlit UI & SHAP visualization
│   ├── requirements.txt
│   └── Dockerfile
│
├── notebook/
│   └── Credit_Card_Fraud_Detection.ipynb   # Full training & benchmarking pipeline
│
├── dataset/
│   └── README.md                       # Dataset download instructions
│
├── docker-compose.yml
└── README.md
```

---

##  Running Locally

**Clone the repository**

```bash
git clone https://github.com/eklavya072/FraudShield-Intelligence.git
cd FraudShield-Intelligence
```

**Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

**Frontend** *(in a separate terminal)*

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

##  Docker

Run the full stack with a single command:

```bash
docker compose up --build
```

---

##  Dataset

This project uses the **PaySim** synthetic financial transaction dataset — a simulation of mobile money transactions designed for fraud detection research.

Due to GitHub file size limitations, the dataset is not included in this repository. Download it from Kaggle and place it inside the `dataset/` directory before running the training notebook.

---

##  Explainable AI with SHAP

FraudShield Intelligence integrates **SHAP (SHapley Additive exPlanations)** to provide per-prediction transparency. For every transaction evaluated, SHAP computes the contribution of each individual feature to the final prediction — making the model auditable and interpretable rather than a black box.

This is particularly important in financial applications, where regulators and end users require justification for risk classifications.

---



