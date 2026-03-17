# �️ VaultEye — Credit Card Fraud Detection System

An end-to-end machine learning project that detects fraudulent credit card transactions using **Random Forest classification**, **PCA transformation**, and a premium **Streamlit dashboard** with user authentication.

---

## 🎯 Project Description

Financial fraud costs billions annually. **VaultEye** tackles this by building an intelligent detection system that analyzes credit card transactions and classifies them as legitimate or fraudulent in real-time. The project covers the full ML lifecycle:

1. **Data Ingestion** — Load and explore 284,807 real-world transactions
2. **Preprocessing** — Feature scaling (StandardScaler), train/test split with stratification
3. **Class Balancing** — SMOTE oversampling to handle extreme 577:1 class imbalance
4. **PCA Pipeline** — Fit PCA on features and support raw-to-V-component transformation
5. **Model Training** — Random Forest classifier optimized for fraud detection
6. **Evaluation** — Confusion matrix, ROC-AUC, Precision-Recall curves, feature importance
7. **Deployment** — Interactive Streamlit web app with login/register, dashboard, and live prediction

### Key Highlights

| Feature | Details |
|---------|---------|
| **Model** | Random Forest (99.90% accuracy, 0.98 ROC-AUC) |
| **Dataset** | 284,807 transactions, 0.17% fraud rate |
| **PCA Raw Mode** | Transform raw features (F1–F28) into V-components via PCA |
| **Authentication** | User login/register system with password hashing |
| **Dashboard** | Real-time KPIs, system status, evaluation charts |
| **Prediction** | Live fraud probability scoring with adjustable threshold |

---

## 📊 Dataset

- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions over 2 days
- **Features**: 30 numerical features (V1–V28 from PCA, Time, Amount)
- **Target**: `Class` (0 = Legitimate, 1 = Fraud)
- **Imbalance**: Only 492 frauds out of 284,807 transactions (~0.17%)

---

## 🛠️ Skills & Technologies

| Category | Implementation |
|----------|---------------|
| **Data Preprocessing** | Feature scaling (StandardScaler), missing value checks, duplicate analysis |
| **Handling Imbalanced Data** | SMOTE (Synthetic Minority Oversampling Technique) |
| **Dimensionality Reduction** | PCA (Principal Component Analysis) — fit, transform, inverse-transform |
| **Model Training** | Random Forest Classifier with hyperparameter tuning |
| **Model Evaluation** | Confusion Matrix, ROC-AUC, Precision-Recall, Feature Importance |
| **Model Persistence** | joblib serialization (models, scalers, PCA) |
| **Web Application** | Streamlit dashboard with glassmorphism UI |
| **Authentication** | SHA-256 + salt password hashing, JSON user storage |
| **Data Visualization** | matplotlib, seaborn — charts, heatmaps, distribution plots |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`:
```
data/creditcard.csv
```

### 3. Run the ML Pipeline

```bash
python main.py
```

This will:
- Load and explore the dataset
- Preprocess, scale, and apply SMOTE
- Fit and save the PCA model
- Train the Random Forest model
- Evaluate and save results to `plots/`

### 4. Generate Raw Data (Optional)

```bash
python generate_raw_data.py
```

Creates `data/creditcard_raw.csv` by inverse-transforming PCA features — enables the app's **Raw Mode** for PCA demonstration.

### 5. Launch the Web App

```bash
streamlit run app.py
```

1. **Register** a new account or **Sign In**
2. Explore the **Executive Dashboard** with KPIs and charts
3. Go to **Predict Transaction** to run live fraud detection
4. Check **Model Analysis** for performance deep-dives

---

## 📁 Project Structure

```
ML PROJECT 1/
├── data/
│   ├── creditcard.csv           # Original Kaggle dataset
│   └── creditcard_raw.csv       # Raw features (PCA inverse-transformed)
├── models/
│   ├── random_forest.pkl        # Trained RF model
│   ├── scaler_amount.pkl        # Amount scaler
│   ├── scaler_time.pkl          # Time scaler
│   └── pca_transformation.pkl   # Fitted PCA model
├── plots/                       # Generated evaluation plots
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   └── feature_importance_random_forest.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & exploration
│   ├── preprocessing.py         # Scaling, splitting, SMOTE, PCA
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation & visualization
│   └── utils.py                 # Helper functions & paths
├── app.py                       # Streamlit web application
├── auth.py                      # User authentication module
├── main.py                      # End-to-end training pipeline
├── generate_raw_data.py         # Raw data generation script
├── requirements.txt
└── README.md
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.90% |
| **Precision** | 67.21% |
| **Recall (Fraud Catch Rate)** | 83.67% |
| **F1-Score** | 74.55% |
| **ROC-AUC** | 0.9826 |
| **Average Precision** | 0.8496 |

---

## 🖥️ Web Application Features

### Authentication
- **Sign In** and **Register** tabs with password security
- Session-based access control
- User profile display in sidebar

### Executive Dashboard
- Personalized welcome banner with greeting
- 6 KPI metric cards (Total Tx, Legit, Fraud, Rate, Avg Fraud Amount, Model Accuracy)
- System status indicators (Model, PCA, Dataset, Raw Data)
- Interactive charts (class distribution, amount analysis)
- Evaluation plots (confusion matrix, ROC, precision-recall)

### Predict Transaction
- **Standard Mode**: Enter V1–V28 PCA components directly
- **Raw Mode**: Enter F1–F28 raw features → auto-transformed via PCA
- Quick templates: Load real legitimate/fraud samples
- Adjustable classification threshold
- Visual fraud probability result with confidence bar

### Model Analysis
- Benchmark metrics (Accuracy, ROC-AUC, AP, Recall)
- ROC discrimination curve
- Confusion matrix heatmap

---

## 📝 License

This project is for educational purposes.
