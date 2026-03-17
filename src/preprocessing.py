"""
preprocessing.py — Feature scaling, train/test split, and SMOTE resampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, cast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from .utils import print_header


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    """
    Apply StandardScaler to 'Amount' and 'Time' columns.
    Fits ONLY on X_train to prevent data leakage, then transforms both.
    V1–V28 are already PCA-transformed and scaled.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler_amount, scaler_time
    """
    print_header("⚙️  Feature Scaling")

    X_train = X_train.copy()
    X_test = X_test.copy()
    
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    # Fit and transform training data
    X_train["Amount"] = scaler_amount.fit_transform(X_train[["Amount"]])
    X_train["Time"] = scaler_time.fit_transform(X_train[["Time"]])

    # Transform test data (no fitting!)
    X_test["Amount"] = scaler_amount.transform(X_test[["Amount"]])
    X_test["Time"] = scaler_time.transform(X_test[["Time"]])

    print("  ✅ Scaled 'Amount' and 'Time' (Fit on Train only)")
    print(f"     Train Amount — mean: {X_train['Amount'].mean():.4f}, std: {X_train['Amount'].std():.4f}")
    print(f"     Test  Amount — mean: {X_test['Amount'].mean():.4f}, std: {X_test['Amount'].std():.4f}")

    return X_train, X_test, scaler_amount, scaler_time


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:  # type: ignore[type-arg]
    """
    Stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print_header("✂️  Train/Test Split")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    splits: Any = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train = cast(pd.DataFrame, splits[0])
    X_test = cast(pd.DataFrame, splits[1])
    y_train = cast(pd.Series, splits[2])
    y_test = cast(pd.Series, splits[3])

    print(f"  Training set : {X_train.shape[0]:>7,} samples")
    print(f"  Test set     : {X_test.shape[0]:>7,} samples")
    print(f"  Fraud in train: {y_train.sum():>5,}  ({y_train.mean()*100:.4f}%)")
    print(f"  Fraud in test : {y_test.sum():>5,}  ({y_test.mean()*100:.4f}%)")

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to the training set to balance classes.
    IMPORTANT: SMOTE is applied ONLY to training data, never to test data.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
    y_train : pd.Series or np.ndarray
    random_state : int

    Returns
    -------
    X_resampled, y_resampled
    """
    print_header("🔄 SMOTE Oversampling")

    print(f"  Before SMOTE:")
    print(f"    Class 0 (Legitimate): {np.sum(y_train == 0):,}")
    print(f"    Class 1 (Fraud):      {np.sum(y_train == 1):,}")

    smote = SMOTE(random_state=random_state)
    result = smote.fit_resample(X_train, y_train)
    X_resampled, y_resampled = result[0], result[1]

    print(f"\n  After SMOTE:")
    print(f"    Class 0 (Legitimate): {np.sum(y_resampled == 0):,}")
    print(f"    Class 1 (Fraud):      {np.sum(y_resampled == 1):,}")
    print(f"    Total samples:        {len(y_resampled):,}")

    return X_resampled, y_resampled


def fit_pca(df: pd.DataFrame, n_components: int = 28) -> PCA:
    """
    Fit PCA on the features to reduce dimensionality to n_components.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with raw features (excluding Time, Amount, Class)
    n_components : int
        Number of components (default 28)
        
    Returns
    -------
    PCA
        Fitted PCA object
    """
    print_header(f"📉 PCA Fitting ({n_components} components)")
    
    # Exclude non-PCA features
    features = [col for col in df.columns if col not in ["Time", "Amount", "Class"]]
    X = df[features]
    
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  ✅ PCA Fitted. Variance explained: {explained_var:.2f}%")
    
    return pca


def apply_pca(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    """
    Apply a fitted PCA model to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
    pca : PCA
    
    Returns
    -------
    pd.DataFrame
        DataFrame with V1...Vn components + Time, Amount, Class
    """
    # Exclude non-PCA features
    features = [col for col in df.columns if col not in ["Time", "Amount", "Class"]]
    X = df[features]
    
    v_features = pca.transform(X)
    v_cols = [f"V{i+1}" for i in range(v_features.shape[1])]
    
    v_df = pd.DataFrame(v_features, columns=v_cols, index=df.index)
    
    # Add back the other columns if they exist
    for col in ["Time", "Amount", "Class"]:
        if col in df.columns:
            v_df[col] = df[col]
            
    return v_df


def preprocess_pipeline(df: pd.DataFrame, pca_model: PCA | None = None):
    """
    Full preprocessing pipeline: PCA (optional) → split → scale → SMOTE.

    Parameters
    ----------
    df : pd.DataFrame
    pca_model : PCA, optional
        If provided, applies this PCA model. If None and V1-V28 are missing,
        it should ideally fit PCA, but for this project we assume PCA 
        is either pre-applied or handled externally.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test,
                    X_train_smote, y_train_smote,
                    scaler_amount, scaler_time, pca
    """
    # 0. Apply PCA if model is provided or if we detect "Raw" data
    # (Simplified for this project: if V1 isn't there, we'd need PCA)
    current_df = df.copy()
    fitted_pca = pca_model
    
    if "V1" not in df.columns and pca_model is None:
        fitted_pca = fit_pca(df)
        current_df = apply_pca(df, fitted_pca)
    elif pca_model is not None:
        current_df = apply_pca(df, pca_model)

    # 1. Split (Current Data)
    X_train_raw, X_test_raw, y_train, y_test = split_data(current_df)

    # 2. Scale (Fit on X_train only)
    X_train, X_test, scaler_amount, scaler_time = scale_features(X_train_raw, X_test_raw)

    # 3. SMOTE on training data only
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "scaler_amount": scaler_amount,
        "scaler_time": scaler_time,
        "pca": fitted_pca
    }
