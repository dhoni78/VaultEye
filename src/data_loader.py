"""
data_loader.py — Load and explore the Credit Card Fraud dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

from .utils import DATASET_PATH, RAW_DATASET_PATH, PLOTS_DIR, ensure_dirs, print_header



def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the credit card dataset from CSV.
    Automatically detects and extracts ZIP archives (common with Kaggle downloads).

    Parameters
    ----------
    path : str
        Path to creditcard.csv

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {path}\n"
            "   Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "   Place the file at: data/creditcard.csv\n"
        )

    # Auto-detect if the file is actually a ZIP archive (common with Kaggle downloads)
    if zipfile.is_zipfile(path):
        print("  ⚠️  Dataset file is a ZIP archive — extracting automatically...")
        data_dir = os.path.dirname(path)
        with zipfile.ZipFile(path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV file found inside the ZIP archive.")
            zf.extract(csv_names[0], data_dir)
            extracted_path = os.path.join(data_dir, csv_names[0])
        # Replace the ZIP with the extracted CSV
        if extracted_path != path:
            os.replace(extracted_path, path)
        else:
            # The ZIP was overwritten by the extract; nothing extra to do
            pass
        print(f"  ✅ Extracted '{csv_names[0]}' from ZIP archive")

    df = pd.read_csv(path)
    print(f"  ✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_raw_data(path: str = RAW_DATASET_PATH) -> pd.DataFrame:
    """
    Load the raw (inverse-PCA) credit card dataset.

    Parameters
    ----------
    path : str
        Path to creditcard_raw.csv

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ Raw dataset not found at: {path}\n"
            "   Run: python generate_raw_data.py\n"
        )

    df = pd.read_csv(path)
    print(f"  ✅ Raw dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Print exploratory statistics and return a summary dict.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict  — summary statistics
    """
    print_header("📊 Data Exploration")

    # Basic info
    print("Shape:", df.shape)
    print("\nColumn Types:")
    print(df.dtypes.value_counts().to_string())
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Class distribution
    class_counts = df["Class"].value_counts()
    class_pct = df["Class"].value_counts(normalize=True) * 100

    print("\n── Class Distribution ──")
    print(f"  Legitimate (0): {class_counts[0]:>8,}  ({class_pct[0]:.4f}%)")
    print(f"  Fraudulent (1): {class_counts[1]:>8,}  ({class_pct[1]:.4f}%)")
    print(f"  Imbalance Ratio: 1 : {class_counts[0] // class_counts[1]}")

    # Descriptive stats for key columns
    print("\n── Amount Statistics ──")
    print(df["Amount"].describe().to_string())

    print("\n── Time Statistics ──")
    print(df["Time"].describe().to_string())

    return {
        "shape": df.shape,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_counts": class_counts.to_dict(),
        "class_pct": class_pct.to_dict(),
    }


def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    """
    Plot the class distribution as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
    save : bool
        If True, save the plot to plots/
    """
    ensure_dirs()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    class_counts = df["Class"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Legitimate", "Fraudulent"]

    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Transaction Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=12)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontsize=11, fontweight="bold")

    # Amount distribution by class
    legitimate = df[df["Class"] == 0]["Amount"]
    fraudulent = df[df["Class"] == 1]["Amount"]

    axes[1].hist(legitimate, bins=50, alpha=0.7, label="Legitimate", color="#2ecc71", edgecolor="white")
    axes[1].hist(fraudulent, bins=50, alpha=0.7, label="Fraudulent", color="#e74c3c", edgecolor="white")
    axes[1].set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Amount ($)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_xlim(0, 500)  # Zoom in for visibility
    axes[1].legend(fontsize=11)

    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved → {path}")

    plt.close(fig)
