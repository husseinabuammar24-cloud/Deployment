import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
)

"""
train_model.py

Full pipeline in one file: clean_data -> preprocess_data ->
train_and_evaluate_all -> save best model.

Run directly:
    python train_model.py --data Data/SaleCleanForAnalysis.csv
"""

MODELS_DIR = "models"

TARGET = "price"

DROP_COLS = [
    "seller_id", "street", "postal_code", "transaction_type",
    "garden", "energy_consumption", "street_number",
    "garage", "terrace",
]
BINARY_COLS = ["balcony", "swimming_pool", "elevator", "furnished"]
NUMERIC_COLS = [
    "livable_surface", "number_of_bedrooms", "number_of_bathrooms",
    "land_surface", "date_of_construction",
]
CATEGORICAL_COLS = [
    "property_type", "property_subtype", "property_condition",
    "province", "availability_clean",
]
DUPLICATE_SUBSET_COLS = ["latitude", "longitude", "price", "livable_surface"]
OUTLIER_COLS = ["price", "livable_surface", "land_surface"]
NUMERIC_FULL = NUMERIC_COLS + ["latitude", "longitude"]


# ---------------------------------------------------------------------------
# 1. Clean data
# ---------------------------------------------------------------------------

def simplify_availability(val):
    """Collapse raw `availability` values into 5 clean buckets."""
    if pd.isna(val):
        return "Unknown"
    if val in ["On contract", "Immediately", "Negotiable"]:
        return val
    if str(val).startswith("From"):
        return "Future date"
    return val


def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Drop rows where any of `columns` falls outside the IQR fence."""
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns:
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        print(f"{col}: removed {before - len(df_clean)} rows (kept range {lower:,.0f} - {upper:,.0f})")
    return df_clean


def clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and structurally clean the raw listings CSV. This step does NOT
    impute numeric/categorical gaps with statistics (median/mode) — that
    happens in `preprocess_data`, fit on TRAIN ONLY, so the same fitted
    values can be reused on new data in `predict.py` without leakage.

      1. Drop exact duplicate rows.
      2. Drop redundant / high-missingness columns.
      3. Fill true binary flags (missing = False, "not present").
      4. Simplify `availability` -> `availability_clean`.
      5. Drop rows still missing latitude/longitude/price/livable_surface
         (these can't be sensibly imputed).
      6. Re-run dedup (column drops/imputing can reveal new duplicates).
      7. Remove outliers (IQR method) on price / livable_surface / land_surface.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {filepath} -> shape {df.shape}")

    df = df.drop_duplicates()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    binary_cols_present = [c for c in BINARY_COLS if c in df.columns]
    df[binary_cols_present] = df[binary_cols_present].fillna(False).astype(int)

    if "availability" in df.columns:
        df["availability_clean"] = df["availability"].apply(simplify_availability)
        df = df.drop(columns=["availability"])

    subset_cols = [c for c in DUPLICATE_SUBSET_COLS if c in df.columns]
    df = df.dropna(subset=subset_cols)

    df = df.drop_duplicates()

    #df = remove_outliers_iqr(df, OUTLIER_COLS)

    print(f"Cleaned shape: {df.shape} | Missing values: {df.isna().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocess data
# ---------------------------------------------------------------------------

def fit_impute_values(X_train: pd.DataFrame) -> dict:
    """
    Compute imputation values from TRAIN ONLY:
      - numeric columns -> median
      - categorical columns -> "Unknown"
    Returned dict is saved as an artifact so predict.py can apply the exact
    same fill values to brand-new listings.
    """
    impute_values = {}
    for col in NUMERIC_COLS:
        if col in X_train.columns:
            impute_values[col] = X_train[col].median()
    for col in CATEGORICAL_COLS:
        if col in X_train.columns:
            impute_values[col] = "Unknown"
    return impute_values


def apply_impute_values(df: pd.DataFrame, impute_values: dict) -> pd.DataFrame:
    """Fill numeric/categorical NaNs using previously-fit impute_values."""
    df = df.copy()
    for col, value in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


def preprocess_data(df: pd.DataFrame, target: str = TARGET, test_size: float = 0.2, random_state: int = 42):
    """
    Split cleaned data, impute remaining gaps (fit on TRAIN only), then
    one-hot encode categoricals and scale numerics (also fit on TRAIN only)
    to avoid leakage.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler, impute_values, feature_names
    """
    df = df.copy()
    df[BINARY_COLS] = df[BINARY_COLS].astype(int)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Split shapes:", X_train.shape, X_test.shape)

    # Impute remaining numeric/categorical gaps — fit on TRAIN only
    impute_values = fit_impute_values(X_train)
    X_train = apply_impute_values(X_train, impute_values)
    X_test = apply_impute_values(X_test, impute_values)

    # One-hot encode — fit on TRAIN only
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[CATEGORICAL_COLS])

    def encode(frame):
        encoded_array = encoder.transform(frame[CATEGORICAL_COLS])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
            index=frame.index,
        )
        return pd.concat([frame.drop(columns=CATEGORICAL_COLS), encoded_df], axis=1)

    X_train_encoded = encode(X_train)
    X_test_encoded = encode(X_test)

    # Scale numeric columns — fit on TRAIN only
    scaler = StandardScaler()
    scaler.fit(X_train_encoded[NUMERIC_FULL])

    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    X_train_scaled[NUMERIC_FULL] = scaler.transform(X_train_encoded[NUMERIC_FULL])
    X_test_scaled[NUMERIC_FULL] = scaler.transform(X_test_encoded[NUMERIC_FULL])

    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)

    feature_names = list(X_train_scaled.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler, impute_values, feature_names


# ---------------------------------------------------------------------------
# 3. Train model(s)
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


TRAINERS = {
    "Linear Regression": train_linear_regression,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
}


# ---------------------------------------------------------------------------
# 4. Predict (thin wrapper — predict.py handles new/unseen data end-to-end)
# ---------------------------------------------------------------------------

def predict(model, X):
    return model.predict(X)


# ---------------------------------------------------------------------------
# 5. Evaluate model
# ---------------------------------------------------------------------------

def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Score a fitted model on train/test and flag possible overfitting."""
    y_pred_train = predict(model, X_train)
    y_pred_test = predict(model, X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    gap = train_r2 - test_r2

    print(f"--- {name} ---")
    print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} (gap: {gap:.4f})")
    print(f"MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | MAPE: {mape:.2f}%")
    print("Overfitting:", "Yes, gap is large" if gap > 0.1 else "No significant overfitting")
    print()

    return {
        "model": name, "train_r2": train_r2, "test_r2": test_r2,
        "mae": mae, "rmse": rmse, "mape": mape,
    }


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Train Linear Regression, Random Forest and XGBoost, evaluate each,
    and return (results_df, fitted_models) so the caller can pick a winner.
    """
    fitted_models = {}
    results = []

    for name, trainer in TRAINERS.items():
        model = trainer(X_train, y_train)
        fitted_models[name] = model
        results.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    return results_df, fitted_models


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_artifacts(model, encoder, scaler, impute_values, feature_names, models_dir: str = MODELS_DIR):
    """
    Pickle the fitted model together with everything predict.py needs to
    reproduce identical feature transformations on brand-new data:
    the encoder, the scaler, the training-time imputation values, and the
    exact training-time column order (feature_names) — models like XGBoost
    require new data's columns to match this order exactly, not just match
    by name.
    """
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "impute_values.pkl"), "wb") as f:
        pickle.dump(impute_values, f)
    with open(os.path.join(models_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print(f"Saved model.pkl, encoder.pkl, scaler.pkl, impute_values.pkl, feature_names.pkl -> {models_dir}/")


def load_artifacts(models_dir: str = MODELS_DIR):
    """
    Reads the five files that save_artifacts previously wrote to disk —
    model.pkl, encoder.pkl, scaler.pkl, impute_values.pkl, feature_names.pkl —
    and reconstructs them back into live Python objects in memory.
    """
    with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "impute_values.pkl"), "rb") as f:
        impute_values = pickle.load(f)
    with open(os.path.join(models_dir, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    return model, encoder, scaler, impute_values, feature_names


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(data_path: str, models_dir: str = "models"):
    """
    Run the full training pipeline end-to-end: clean -> preprocess -> train ->
    evaluate -> select and persist the best model.

    Steps
    -----
    1. Load and clean the raw listings CSV (`clean_data`).
    2. Split into train/test, impute remaining gaps, and fit-transform
       categorical/numeric features (`preprocess_data`).
    3. Train Linear Regression, Random Forest, and XGBoost; evaluate each on
       train/test (`train_and_evaluate_all`).
    4. Pick the model with the highest test R² as the winner.
    5. Save the winning model together with the encoder/scaler/impute_values
       it depends on (`save_artifacts`), so `predict.py` can reproduce
       identical feature transformations on new data.

    Parameters
    ----------
    data_path : str
        Path to the raw listings CSV (e.g. "Data/SaleCleanForAnalysis.csv").
    models_dir : str, default "models"
        Folder to save model.pkl, encoder.pkl, scaler.pkl, impute_values.pkl into.

    Returns
    -------
    results_df : pd.DataFrame
        Comparison of all trained models' metrics, sorted by test R² (best first).
    best_model : estimator
        The fitted model with the highest test R² (already saved to disk).
    """

    # 1. Clean
    df = clean_data(data_path)

    # 2. Preprocess (split + impute + encode + scale)
    X_train, X_test, y_train, y_test, encoder, scaler, impute_values, feature_names = preprocess_data(df)

    # 3-5. Train + evaluate Linear Regression, Random Forest, XGBoost
    results_df, fitted_models = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    print("\n=== Model comparison (sorted by test R²) ===")
    print(results_df.to_string(index=False))

    # Pick the winner and persist it alongside everything it needs
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (test R² = {results_df.iloc[0]['test_r2']:.4f})")

    save_artifacts(best_model, encoder, scaler, impute_values, feature_names, models_dir=models_dir)

    return results_df, best_model


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Immo Eliza price models.")
    parser.add_argument("--data", default="Data/SaleCleanForAnalysis.csv", help="Path to the raw listings CSV.")
    parser.add_argument("--models-dir", default="models", help="Where to save model.pkl/encoder.pkl/scaler.pkl/impute_values.pkl.")
    args = parser.parse_args()

    run(args.data, models_dir=args.models_dir)


if __name__ == "__main__":
    main()