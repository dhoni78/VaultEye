"""
main.py — End-to-end Credit Card Fraud Detection pipeline.

Usage:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

# Suppress harmless ChildProcessError from multiprocessing resource tracker
# during process cleanup (Python 3.13 known issue)
from multiprocessing.resource_tracker import ResourceTracker as _RT
_original_stop = _RT._stop  # type: ignore[attr-defined]

def _patched_stop(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    try:
        _original_stop(self, *args, **kwargs)
    except ChildProcessError:
        pass

_RT._stop = _patched_stop  # type: ignore[attr-defined]

from src.data_loader import load_data, explore_data, plot_class_distribution
from src.preprocessing import preprocess_pipeline, fit_pca
from src.train import train_all_models
from src.evaluate import evaluate_all_models
from src.utils import print_header, save_scaler, save_pca


def main():
    print_header("💳 Credit Card Fraud Detection System")
    print("  End-to-End ML Pipeline\n")

    # ── Step 1: Load Data ────────────────────────────────────────────────────
    print_header("📂 Step 1 — Loading Data")
    df = load_data()

    # ── Step 2: Explore Data ─────────────────────────────────────────────────
    summary = explore_data(df)
    plot_class_distribution(df)

    # ── Step 3: Preprocess ───────────────────────────────────────────────────
    print_header("📂 Step 3 — Preprocessing")
    data = preprocess_pipeline(df)
    
    # Save scalers for inference
    save_scaler(data["scaler_amount"], "amount")
    save_scaler(data["scaler_time"], "time")
    
    # Always fit & save a PCA model so the app's Raw Mode can use it.
    # If preprocess_pipeline already fitted one (raw data), use that.
    # Otherwise, fit one from the existing V-features.
    if data["pca"] is not None:
        save_pca(data["pca"], "transformation")
    else:
        # Dataset already has V1-V28; fit PCA on those for the Raw Mode demo
        pca_model = fit_pca(df)
        save_pca(pca_model, "transformation")

    # ── Step 4: Train Models ─────────────────────────────────────────────────
    trained_models = train_all_models(
        data["X_train_smote"],
        data["y_train_smote"],
    )

    # ── Step 5: Evaluate Models ──────────────────────────────────────────────
    feature_names = list(data["X_test"].columns)
    all_results = evaluate_all_models(
        trained_models,
        data["X_test"],
        data["y_test"],
        feature_names=feature_names,
    )

    # ── Done ─────────────────────────────────────────────────────────────────
    print_header("✅ Pipeline Complete!")
    print("  Trained models saved to    → models/")
    print("  Evaluation plots saved to  → plots/")
    print("  Run the Streamlit app:       streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
