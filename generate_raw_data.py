"""
generate_raw_data.py — Create a synthetic "raw" dataset by inverting the
existing PCA transformation on V1–V28.

This produces `data/creditcard_raw.csv` with columns:
    Time, F1, F2, …, F28, Amount, Class

Usage:
    python generate_raw_data.py
"""

import os
import numpy as np
import pandas as pd
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

INPUT_CSV = os.path.join(DATA_DIR, "creditcard.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "creditcard_raw.csv")
PCA_MODEL = os.path.join(MODELS_DIR, "pca_transformation.pkl")


def main():
    # ── 1. Validate pre-requisites ────────────────────────────────────────────
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Dataset not found at {INPUT_CSV}\n"
            "  Run main.py first (or place creditcard.csv in data/)."
        )
    if not os.path.exists(PCA_MODEL):
        raise FileNotFoundError(
            f"PCA model not found at {PCA_MODEL}\n"
            "  Run main.py first so the PCA model is fitted and saved."
        )

    # ── 2. Load dataset & PCA model ───────────────────────────────────────────
    print("📂 Loading creditcard.csv …")
    df = pd.read_csv(INPUT_CSV)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("📂 Loading PCA model …")
    pca = joblib.load(PCA_MODEL)
    print(f"   Components: {pca.n_components_}, Features in: {pca.n_features_in_}")

    # ── 3. Inverse-transform V1–V28 → F1–F28 ─────────────────────────────────
    v_cols = [f"V{i}" for i in range(1, 29)]
    V_data = df[v_cols].values

    print("🔄 Inverse-transforming V1–V28 → F1–F28 …")
    raw_features = pca.inverse_transform(V_data)

    f_cols = [f"F{i}" for i in range(1, raw_features.shape[1] + 1)]
    raw_df = pd.DataFrame(raw_features, columns=f_cols)

    # ── 4. Attach Time, Amount, Class ─────────────────────────────────────────
    raw_df.insert(0, "Time", df["Time"].values)
    raw_df["Amount"] = df["Amount"].values
    raw_df["Class"] = df["Class"].values

    print(f"   Raw dataset shape: {raw_df.shape}")

    # ── 5. Round-trip validation ──────────────────────────────────────────────
    print("✅ Round-trip validation …")
    reconstructed_v = pca.transform(raw_features)
    max_error = np.max(np.abs(V_data - reconstructed_v))
    mean_error = np.mean(np.abs(V_data - reconstructed_v))
    print(f"   Max absolute error:  {max_error:.2e}")
    print(f"   Mean absolute error: {mean_error:.2e}")

    if max_error > 1e-6:
        print("   ⚠️  Error is larger than expected — results may vary slightly.")
    else:
        print("   ✅ Round-trip is near-perfect!")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print(f"💾 Saving → {OUTPUT_CSV}")
    raw_df.to_csv(OUTPUT_CSV, index=False)
    size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"   Done! ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
