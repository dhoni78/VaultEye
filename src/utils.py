"""
utils.py — Helper functions for the Credit Card Fraud Detection System.
"""

import os
import time
import joblib
import numpy as np


# ─── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

DATASET_PATH = os.getenv("FRAUD_DATASET_PATH", os.path.join(DATA_DIR, "creditcard.csv"))
RAW_DATASET_PATH = os.path.join(DATA_DIR, "creditcard_raw.csv")


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [MODELS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ─── Model Persistence ──────────────────────────────────────────────────────────

def save_model(model, name: str):
    """Save a trained model to the models/ directory."""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  ✅ Model saved → {path}")
    return path


def load_model(name: str):
    """Load a saved model from the models/ directory."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ─── Scaler Persistence ─────────────────────────────────────────────────────────

def save_scaler(scaler, name: str):
    """Save a fitted scaler to the models/ directory."""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"scaler_{name}.pkl")
    joblib.dump(scaler, path)
    print(f"  ✅ Scaler saved → {path}")
    return path


def load_scaler(name: str):
    """Load a saved scaler from the models/ directory."""
    path = os.path.join(MODELS_DIR, f"scaler_{name}.pkl")
    if not os.path.exists(path):
        # Fallback for app.py if it hasn't been run yet
        return None
    return joblib.load(path)


# ─── PCA Persistence ────────────────────────────────────────────────────────────

def save_pca(pca, name: str):
    """Save a fitted PCA model to the models/ directory."""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, f"pca_{name}.pkl")
    joblib.dump(pca, path)
    print(f"  ✅ PCA model saved → {path}")
    return path


def load_pca(name: str):
    """Load a saved PCA model from the models/ directory."""
    path = os.path.join(MODELS_DIR, f"pca_{name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─── Timer ───────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ─── Formatting ─────────────────────────────────────────────────────────────────

def print_header(title: str):
    """Print a styled section header."""
    line = "═" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")
