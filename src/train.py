"""
train.py — Train multiple classifiers for fraud detection.
"""

from sklearn.ensemble import RandomForestClassifier

from .utils import save_model, Timer, print_header


# ─── Model Definitions ──────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dictionary containing only the most effective model.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
        ),
    }


# ─── Pretty Names ───────────────────────────────────────────────────────────────

MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
}


# ─── Training ───────────────────────────────────────────────────────────────────

def train_single_model(name, model, X_train, y_train):
    """
    Train a single model and save it.

    Parameters
    ----------
    name : str
    model : sklearn/xgboost estimator
    X_train : array-like
    y_train : array-like

    Returns
    -------
    model, elapsed_time (str)
    """
    display = MODEL_DISPLAY_NAMES.get(name, name)
    print(f"\n  🏋️  Training {display}...")

    with Timer() as t:
        model.fit(X_train, y_train)

    print(f"     ⏱ Training time: {t}")
    save_model(model, name)

    return model, str(t)


def train_all_models(X_train, y_train) -> dict:
    """
    Train all models and return results.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like

    Returns
    -------
    dict  — {model_name: {"model": fitted_model, "time": elapsed_time}}
    """
    print_header("🚀 Model Training")

    models = get_models()
    results = {}

    for name, model in models.items():
        fitted_model, elapsed = train_single_model(name, model, X_train, y_train)
        results[name] = {
            "model": fitted_model,
            "time": elapsed,
        }

    print(f"\n  ✅ All {len(results)} models trained and saved!")
    return results
