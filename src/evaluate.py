"""
evaluate.py — Model evaluation: confusion matrix, ROC-AUC, precision-recall, comparison table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from .utils import PLOTS_DIR, ensure_dirs, print_header
from .train import MODEL_DISPLAY_NAMES


# ─── Single Model Evaluation ────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single model and return metrics.

    Parameters
    ----------
    model : fitted estimator
    X_test : array-like
    y_test : array-like
    model_name : str

    Returns
    -------
    dict  — metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0.0),
        "recall": recall_score(y_test, y_pred, zero_division=0.0),
        "f1": f1_score(y_test, y_pred, zero_division=0.0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    print(f"\n  📋 {display}")
    print(f"     Accuracy:   {metrics['accuracy']:.4f}")
    print(f"     Precision:  {metrics['precision']:.4f}")
    print(f"     Recall:     {metrics['recall']:.4f}")
    print(f"     F1-Score:   {metrics['f1']:.4f}")
    print(f"     ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"     Avg Prec:   {metrics['avg_precision']:.4f}")

    return metrics


# ─── Confusion Matrix Plot ───────────────────────────────────────────────────────

def plot_confusion_matrices(all_results: dict, y_test, save: bool = True):
    """
    Plot confusion matrices for all models in a single figure.

    Parameters
    ----------
    all_results : dict  — {model_name: {"metrics": {...}}}
    y_test : array-like
    save : bool
    """
    ensure_dirs()
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, all_results.items()):
        y_pred = data["metrics"]["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        display = MODEL_DISPLAY_NAMES.get(name, name)

        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax, cbar=False,
            annot_kws={"size": 14, "fontweight": "bold"},
        )
        ax.set_title(display, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)

    plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved → {path}")

    plt.close(fig)


# ─── ROC Curve ───────────────────────────────────────────────────────────────────

def plot_roc_curves(all_results: dict, y_test, save: bool = True):
    """
    Plot ROC curves for all models overlaid.

    Parameters
    ----------
    all_results : dict
    y_test : array-like
    save : bool
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for (name, data), color in zip(all_results.items(), colors):
        y_proba = data["metrics"]["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = data["metrics"]["roc_auc"]
        display = MODEL_DISPLAY_NAMES.get(name, name)

        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{display} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "roc_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved → {path}")

    plt.close(fig)


# ─── Precision-Recall Curve ──────────────────────────────────────────────────────

def plot_precision_recall_curves(all_results: dict, y_test, save: bool = True):
    """
    Plot precision-recall curves for all models.

    Parameters
    ----------
    all_results : dict
    y_test : array-like
    save : bool
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for (name, data), color in zip(all_results.items(), colors):
        y_proba = data["metrics"]["y_proba"]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = data["metrics"]["avg_precision"]
        display = MODEL_DISPLAY_NAMES.get(name, name)

        ax.plot(recall, precision, color=color, lw=2.5, label=f"{display} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved → {path}")

    plt.close(fig)


# ─── Feature Importance (for tree-based models) ─────────────────────────────────

def plot_feature_importance(model, feature_names, model_name: str, top_n: int = 15, save: bool = True):
    """
    Plot top N feature importances for tree-based models.

    Parameters
    ----------
    model : fitted estimator with feature_importances_
    feature_names : list
    model_name : str
    top_n : int
    save : bool
    """
    if not hasattr(model, "feature_importances_"):
        return

    ensure_dirs()
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(indices)),
        importances[indices],
        color="#3498db",
        edgecolor="white",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {display}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Plot saved → {path}")

    plt.close(fig)


# ─── Comparison Table ────────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict) -> pd.DataFrame:
    """
    Print a formatted comparison table of all model metrics.

    Parameters
    ----------
    all_results : dict

    Returns
    -------
    pd.DataFrame
    """
    print_header("📊 Model Comparison Summary")

    rows = []
    for name, data in all_results.items():
        m = data["metrics"]
        rows.append({
            "Model": MODEL_DISPLAY_NAMES.get(name, name),
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Avg Precision": f"{m['avg_precision']:.4f}",
            "Train Time": data.get("time", "—"),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ─── Full Evaluation Pipeline ───────────────────────────────────────────────────

def evaluate_all_models(trained_models: dict, X_test, y_test, feature_names=None) -> dict:
    """
    Evaluate all trained models and generate all plots.

    Parameters
    ----------
    trained_models : dict  — {name: {"model": ..., "time": ...}}
    X_test : array-like
    y_test : array-like
    feature_names : list, optional

    Returns
    -------
    dict  — {name: {"model": ..., "metrics": ..., "time": ...}}
    """
    print_header("🔍 Model Evaluation")

    all_results = {}
    for name, data in trained_models.items():
        model = data["model"]
        metrics = evaluate_model(model, X_test, y_test, name)
        all_results[name] = {
            "model": model,
            "metrics": metrics,
            "time": data["time"],
        }

    # Generate plots
    print("\n  Generating evaluation plots...")
    plot_confusion_matrices(all_results, y_test)
    plot_roc_curves(all_results, y_test)
    plot_precision_recall_curves(all_results, y_test)

    # Feature importance for Random Forest
    if feature_names is not None and "random_forest" in all_results:
        plot_feature_importance(
            all_results["random_forest"]["model"],
            feature_names,
            "random_forest",
        )

    # Comparison table
    comparison_df = print_comparison_table(all_results)
    all_results["_comparison_df"] = comparison_df

    return all_results
