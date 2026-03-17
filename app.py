"""
app.py — Streamlit Web App for Credit Card Fraud Detection.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
)

from src.utils import MODELS_DIR, PLOTS_DIR, DATASET_PATH, RAW_DATASET_PATH, load_scaler, load_pca
from src.train import MODEL_DISPLAY_NAMES
from auth import register_user, authenticate_user, get_user

# ─── Page Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VaultEye | Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">

<style>
    :root {
        --bg-dark: #0a0b10;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --accent-primary: #6366f1;
        --accent-secondary: #a855f7;
        --accent-success: #10b981;
        --accent-danger: #ef4444;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Main App Overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617);
        font-family: 'Outfit', sans-serif;
        color: var(--text-main);
    }

    [data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }

    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 0;
    }

    .sub-header {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Glass Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
    }

    /* Metric Cards */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .metric-label {
        color: var(--text-muted);
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    /* Sidebar Styling */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }

    /* Forms */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* Prediction Cards */
    .result-card {
        padding: 2.5rem;
        border-radius: 32px;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid transparent;
        animation: fadeIn 0.5s ease-out;
    }

    .result-card.fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border-color: rgba(239, 68, 68, 0.3);
    }

    .result-card.legit {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border-color: rgba(16, 185, 129, 0.3);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #fff;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stButton>button {
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }

    .stExpander {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px !important;
    }

    /* Auth Page Styles */
    .auth-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 2rem 0;
    }

    .auth-logo {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .auth-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .auth-subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Welcome Banner */
    .welcome-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 24px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        animation: fadeIn 0.6s ease-out;
    }

    .welcome-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #fff;
        margin-bottom: 0.3rem;
    }

    .welcome-date {
        color: var(--text-muted);
        font-size: 0.95rem;
    }

    /* Status Indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    .status-dot.online { background: #10b981; }
    .status-dot.offline { background: #ef4444; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* User badge */
    .user-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .user-badge-name {
        font-weight: 700;
        color: #fff;
        font-size: 0.95rem;
    }

    .user-badge-role {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────────

@st.cache_resource
def load_transformation_models():
    """Load saved scalers and PCA model."""
    return {
        "amount": load_scaler("amount"),
        "time": load_scaler("time"),
        "pca": load_pca("transformation"),
    }


@st.cache_resource
def load_models():
    """Load all saved models."""
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_dataset():
    """Load the dataset for visualizations."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None


@st.cache_data
def load_raw_dataset():
    """Load the raw (inverse-PCA) dataset for Raw Mode."""
    if os.path.exists(RAW_DATASET_PATH):
        return pd.read_csv(RAW_DATASET_PATH)
    return None


def get_feature_names():
    """Return the list of feature names (V1–V28 + Time + Amount)."""
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def get_raw_feature_names():
    """Return the list of raw feature names (F1–F28)."""
    return [f"F{i}" for i in range(1, 29)]


# ─── Session State Init ─────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"


# ─── Auth Pages ──────────────────────────────────────────────────────────────────

def show_auth_page():
    """Display login and register forms as side-by-side tabs."""
    st.markdown('<div class="auth-logo">🛡️</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">VaultEye</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">AI-Powered Fraud Detection Platform</div>', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register"])

    # ── Sign In Tab ──────────────────────────────────────────────────────────
    with tab_login:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### Welcome Back")
            st.caption("Enter your credentials to access the dashboard.")
            username = st.text_input("Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
            submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    success, msg = authenticate_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username.strip().lower()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Register Tab ─────────────────────────────────────────────────────────
    with tab_register:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("register_form", clear_on_submit=False):
            st.markdown("#### Create Your Account")
            st.caption("Join VaultEye to start detecting fraud.")
            full_name = st.text_input("Full Name", placeholder="Enter your full name", key="reg_name")
            reg_username = st.text_input("Username", placeholder="Choose a username (min 3 chars)", key="reg_user")
            reg_password = st.text_input("Password", type="password", placeholder="Choose a password (min 6 chars)", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            submitted = st.form_submit_button("📝 Create Account", use_container_width=True)

            if submitted:
                if not full_name or not reg_username or not reg_password:
                    st.error("Please fill in all fields.")
                elif reg_password != confirm_password:
                    st.error("❌ Passwords do not match.")
                else:
                    success, msg = register_user(reg_username, reg_password, full_name)
                    if success:
                        st.success(f"✅ {msg}")
                        st.session_state.authenticated = True
                        st.session_state.username = reg_username.strip().lower()
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
        st.markdown('</div>', unsafe_allow_html=True)


# ─── Auth Gate ───────────────────────────────────────────────────────────────────

if not st.session_state.authenticated:
    show_auth_page()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATED ZONE — Everything below requires login
# ═══════════════════════════════════════════════════════════════════════════════

models = load_models()

if not models:
    st.error("⚠️ Models not found. Run main.py")
    st.stop()

# Hardcoded to Random Forest as it's the most effective
model_choice = "random_forest"
if model_choice not in models:
    model_choice = list(models.keys())[0]

# ─── Sidebar (Authenticated) ────────────────────────────────────────────────────

st.sidebar.markdown('<div class="sidebar-title">🛡️ VaultEye</div>', unsafe_allow_html=True)

# User profile badge
user_info = get_user(st.session_state.username)
display_name = user_info["full_name"] if user_info else st.session_state.username

st.sidebar.markdown(f"""
<div class="user-badge">
    <div class="user-badge-name">👤 {display_name}</div>
    <div class="user-badge-role">Fraud Analyst</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["Executive Dashboard", "Predict Transaction", "Model Analysis"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ Settings")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.10,
    step=0.01,
)

st.sidebar.markdown("---")

# Logout button
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()

st.sidebar.caption(f"Powered by {MODEL_DISPLAY_NAMES.get(model_choice, model_choice)}")
st.sidebar.caption(f"Threshold: {threshold:.2f}")

# ─── Executive Dashboard ─────────────────────────────────────────────────────────

if page == "Executive Dashboard":
    st.markdown('<div class="main-header">VAULTEYE DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Financial Integrity & Fraud Detection</div>', unsafe_allow_html=True)

    # Welcome Banner
    now = datetime.now()
    hour = now.hour
    greeting = "Good Morning" if hour < 12 else "Good Afternoon" if hour < 17 else "Good Evening"
    date_str = now.strftime("%A, %B %d, %Y")

    st.markdown(f"""
    <div class="welcome-banner">
        <div class="welcome-name">{greeting}, {display_name} 👋</div>
        <div class="welcome-date">📅 {date_str} &nbsp;|&nbsp; 🕐 {now.strftime("%I:%M %p")}</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()

    if df is not None:
        total = len(df)
        frauds = df["Class"].sum()
        legit = total - frauds
        fraud_pct = (frauds / total) * 100
        avg_fraud_amt = df[df["Class"] == 1]["Amount"].mean()
        avg_legit_amt = df[df["Class"] == 0]["Amount"].mean()

        # ── System Status Bar ────────────────────────────────────────────────
        st.markdown('<div class="section-title">⚡ System Status</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            model_ok = "random_forest" in models
            dot = "online" if model_ok else "offline"
            st.markdown(f'<div class="glass-card"><span class="status-dot {dot}"></span> RF Model {"Active" if model_ok else "Missing"}</div>', unsafe_allow_html=True)
        with s2:
            pca_ok = load_pca("transformation") is not None
            dot = "online" if pca_ok else "offline"
            st.markdown(f'<div class="glass-card"><span class="status-dot {dot}"></span> PCA Engine {"Ready" if pca_ok else "Missing"}</div>', unsafe_allow_html=True)
        with s3:
            dot = "online"
            st.markdown(f'<div class="glass-card"><span class="status-dot {dot}"></span> Dataset Loaded</div>', unsafe_allow_html=True)
        with s4:
            raw_ok = os.path.exists(RAW_DATASET_PATH)
            dot = "online" if raw_ok else "offline"
            st.markdown(f'<div class="glass-card"><span class="status-dot {dot}"></span> Raw Data {"Ready" if raw_ok else "Missing"}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI Metrics (6 cards) ────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="glass-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Transactions</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="glass-card"><div class="metric-value">{legit:,}</div><div class="metric-label">Verified Legitimate</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="glass-card"><div class="metric-value" style="background: linear-gradient(90deg, #f87171, #ef4444); -webkit-background-clip: text;">{int(frauds):,}</div><div class="metric-label">Detected Fraud</div></div>', unsafe_allow_html=True)

        m4, m5, m6 = st.columns(3)
        with m4:
            st.markdown(f'<div class="glass-card"><div class="metric-value">{fraud_pct:.3f}%</div><div class="metric-label">Anomalous Rate</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="glass-card"><div class="metric-value">${avg_fraud_amt:,.2f}</div><div class="metric-label">Avg Fraud Amount</div></div>', unsafe_allow_html=True)
        with m6:
            st.markdown(f'<div class="glass-card"><div class="metric-value" style="background: linear-gradient(90deg, #10b981, #34d399); -webkit-background-clip: text;">99.90%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts Section ───────────────────────────────────────────────────
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown('<div class="section-title">📊 Dataset Integrity</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            colors = ["#10b981", "#ef4444"]
            ax.bar(["Legit", "Fraud"], [legit, int(frauds)], color=colors, width=0.6, edgecolor="white", linewidth=0.5)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.grid(axis='y', alpha=0.1)
            st.pyplot(fig)

        with c_right:
            st.markdown('<div class="section-title">💸 Exposure Analysis</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.4, color="#10b981", label="Legitimate")
            ax.hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.8, color="#ef4444", label="Fraudulent")
            ax.set_xlim(0, 500)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.grid(axis='y', alpha=0.1)
            ax.legend(facecolor="#0f172a", edgecolor="none", labelcolor="white", fontsize=8)
            st.pyplot(fig)

        # ── Evaluation Plots ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">📈 Evaluation Reports</div>', unsafe_allow_html=True)

        plot_files = {
            "Confusion Matrices": "confusion_matrices.png",
            "ROC Curves": "roc_curves.png",
            "Precision-Recall Curves": "precision_recall_curves.png",
        }

        cols = st.columns(len(plot_files))
        for col, (title, filename) in zip(cols, plot_files.items()):
            path = os.path.join(PLOTS_DIR, filename)
            with col:
                if os.path.exists(path):
                    st.image(path, caption=title, use_container_width=True)
                else:
                    st.info(f"Run `python main.py` to generate {title}")

    else:
        st.warning("Dataset not found. Place `creditcard.csv` in the `data/` directory.")


# ─── Predict Transaction Page ────────────────────────────────────────────────────

elif page == "Predict Transaction":
    st.markdown('<div class="main-header">ANALYZE TRANSACTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Risk Assessment Engine</div>', unsafe_allow_html=True)

    df = load_dataset()
    df_raw = load_raw_dataset()
    
    # Initialize session state for feature values if not present
    if "form_values" not in st.session_state:
        st.session_state.form_values = {
            "Time": 0.0,
            "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
    if "raw_form_values" not in st.session_state:
        st.session_state.raw_form_values = {
            "Time": 0.0,
            "Amount": 100.0,
            **{f"F{i}": 0.0 for i in range(1, 29)}
        }
    
    # Input Mode Selection
    input_mode = st.radio(
        "Input Mode",
        ["Standard (V-Components)", "Raw (F-Features → PCA)"],
        horizontal=True,
        help="Standard mode uses pre-processed PCA features. Raw mode transforms your raw F1–F28 inputs into V-components via PCA."
    )

    # Load random sample logic — source depends on mode
    is_raw = input_mode == "Raw (F-Features → PCA)"
    sample_source = df_raw if is_raw else df

    if sample_source is not None:
        st.markdown('<div class="section-title">⚡ Quick Templates</div>', unsafe_allow_html=True)
        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            if st.button("🎲 Simulate Verified Legit", use_container_width=True):
                sample = sample_source[sample_source["Class"] == 0].sample(1).iloc[0]
                if is_raw:
                    st.session_state.raw_form_values = sample.to_dict()
                else:
                    st.session_state.form_values = sample.to_dict()
                st.rerun()
        with col_btn2:
            if st.button("🔴 Simulate Known Fraud", use_container_width=True):
                sample = sample_source[sample_source["Class"] == 1].sample(1).iloc[0]
                if is_raw:
                    st.session_state.raw_form_values = sample.to_dict()
                else:
                    st.session_state.form_values = sample.to_dict()
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Data Entry</div>', unsafe_allow_html=True)

    with st.form("prediction_form", border=False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            time_val = st.number_input(
                "Transaction Time (Offset)", 
                value=float(st.session_state.form_values.get("Time", 0.0)), 
                format="%.2f",
                help="Time in seconds since start of dataset."
            )
        with c2:
            amount_val = st.number_input(
                "Transaction Amount ($)", 
                value=float(st.session_state.form_values.get("Amount", 100.0)), 
                format="%.2f",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        
        if input_mode == "Standard (V-Components)":
            with st.expander("🛠️ Advanced Latent Components (V1 – V28)", expanded=False):
                st.caption("PCA-transformed features representing behavior patterns.")
                v_cols = st.columns(4)
                v_values = []
                for i in range(1, 29):
                    col_idx = (i - 1) % 4
                    with v_cols[col_idx]:
                        key = f"V{i}"
                        val = st.number_input(
                            key, 
                            value=float(st.session_state.form_values.get(key, 0.0)), 
                            format="%.4f", 
                            key=f"input_v{i}"
                        )
                        v_values.append(val)
        else:
            with st.expander("📝 Raw Feature Inputs (F1 – F28)", expanded=True):
                st.caption("Enter raw features — they will be transformed into V1–V28 via PCA.")
                r_cols = st.columns(4)
                raw_features = []
                for i in range(1, 29):
                    col_idx = (i - 1) % 4
                    with r_cols[col_idx]:
                        key = f"F{i}"
                        val = st.number_input(
                            key,
                            value=float(st.session_state.raw_form_values.get(key, 0.0)),
                            format="%.4f",
                            key=f"raw_{i}"
                        )
                        raw_features.append(val)
                v_values = []  # Will be calculated on submit
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 RUN SECURITY AUDIT", use_container_width=True)

    if submitted:
        # Load transformations
        transformations = load_transformation_models()
        
        # 1. Handle PCA if in Raw Mode
        if input_mode == "Raw (F-Features → PCA)":
            pca = transformations["pca"]
            if pca is not None:
                raw_vector = np.array([raw_features])  # shape (1, 28)
                v_values = pca.transform(raw_vector)[0].tolist()
                st.success(f"✨ Transformed F1–F28 → {len(v_values)} V-components via PCA")
            else:
                st.error("⚠️ PCA model not found! Run `python main.py` first.")
                st.stop()
        
        # 2. Build feature vector: [Time, V1..V28, Amount]
        raw_full_vector = np.array([[time_val] + v_values + [amount_val]])
        features = raw_full_vector.copy()
        
        # 3. Scale Time and Amount
        if transformations["time"] and transformations["amount"]:
            features[0, 0] = transformations["time"].transform([[time_val]])[0, 0]
            features[0, -1] = transformations["amount"].transform([[amount_val]])[0, 0]
        else:
            st.warning("⚠️ Scalers not found! Predictions may be inaccurate.")

        model = models[model_choice]

        # 3. Predict using threshold
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0]
            fraud_prob = probability[1]
            prediction = 1 if fraud_prob >= threshold else 0
        else:
            prediction = model.predict(features)[0]
            fraud_prob = 1.0 if prediction == 1 else 0.0
            probability = [1.0 - fraud_prob, fraud_prob]

        st.markdown('<div class="section-title">🛡️ Audit Result</div>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card fraud">
                <h2 style="color: #ef4444; margin:0;">🚨 HIGH RISK DETECTED</h2>
                <p style="color: #fca5a5; font-size: 1.2rem;">Fraud Probability: <b>{fraud_prob*100:.2f}%</b></p>
                <div style="margin-top: 1rem; color: #f8fafc; opacity: 0.8;">Action Required: Review account activity and freeze transactions.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card legit">
                <h2 style="color: #10b981; margin:0;">✅ TRANSACTION SECURE</h2>
                <p style="color: #6ee7b7; font-size: 1.2rem;">Fraud Probability: <b>{fraud_prob*100:.2f}%</b></p>
                <div style="margin-top: 1rem; color: #f8fafc; opacity: 0.8;">Status: Transaction passed all security protocols.</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown('<div class="section-header">Confidence</div>', unsafe_allow_html=True)
        fraud_prob = probability[1]
        color = "#ff416c" if fraud_prob > 0.5 else "#38ef7d"
        st.progress(fraud_prob)


elif page == "Model Analysis":
    st.markdown('<div class="main-header">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into AI Precision & Decision Logic</div>', unsafe_allow_html=True)

    df = load_dataset()

    if df is not None and "random_forest" in models:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = df.drop("Class", axis=1).copy()
        y = df["Class"]

        scaler = StandardScaler()
        X["Amount"] = scaler.fit_transform(X[["Amount"]])
        X["Time"] = scaler.fit_transform(X[["Time"]])

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # ─── High-Level Summary
        st.markdown('<div class="section-title">📈 Benchmark Results (Random Forest)</div>', unsafe_allow_html=True)
        
        model = models["random_forest"]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "Accuracy": f"{np.mean(y_pred == y_test)*100:.2f}%",
            "ROC-AUC": f"{roc_auc_score(y_test, y_proba):.4f}",
            "Detection Power (AP)": f"{average_precision_score(y_test, y_proba):.4f}",
            "Recall (Fraud Catch Rate)": f"{recall_score(y_test, y_pred):.4f}"
        }

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        cols = st.columns(4)
        for col, (label, val) in zip(cols, metrics.items()):
            col.metric(label, val)
        st.markdown('</div>', unsafe_allow_html=True)

        # ─── Visual Analysis
        m_left, m_right = st.columns(2)

        with m_left:
            st.markdown('<div class="section-title">📉 Discrimination Curve</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, color="#10b981", lw=2, label="Random Forest")

            ax.plot([0, 1], [0, 1], "--", color="#94a3b8", alpha=0.3, lw=1)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.spines[:].set_visible(False)
            ax.grid(alpha=0.05)
            ax.legend(facecolor="#0f172a", edgecolor="none", labelcolor="white", fontsize=8)
            st.pyplot(fig)

        with m_right:
            st.markdown('<div class="section-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            sns.heatmap(cm, annot=True, fmt=",d", cmap="Greens", cbar=False, ax=ax,
                        annot_kws={"size": 12, "color": "white"})
            ax.spines[:].set_visible(False)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            st.pyplot(fig)

    else:
        st.info("⚠️ Ensure 'creditcard.csv' is in data/ and Random Forest model is trained.")
