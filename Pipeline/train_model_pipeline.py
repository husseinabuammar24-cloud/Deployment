import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
)

"""
train_model.py  (pipeline version)

Everything that used to be 4 separate fitted objects (impute_values.pkl,
encoder.pkl, scaler.pkl, model.pkl) is now ONE sklearn Pipeline, saved as
a single pipeline.pkl. Calling .fit() fits every step; calling .predict()
runs every step in the exact same order, automatically. No more manually
re-applying imputer -> encoder -> scaler by hand in main.py.

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
NUMERIC_FULL = NUMERIC_COLS + ["latitude", "longitude"]


# ---------------------------------------------------------------------------
# 1. Clean data — structural cleaning only. This stays OUTSIDE the sklearn
#    Pipeline because it's row-level logic with no parameters to "fit"
#    (dropping columns, renaming, collapsing string categories) rather than
#    a statistical transform like imputation/scaling that must be fit on
#    train data only.
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


def clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and structurally clean the raw listings CSV.

      1. Drop exact duplicate rows.
      2. Drop redundant / high-missingness columns.
      3. Fill true binary flags (missing = False, "not present").
      4. Simplify `availability` -> `availability_clean`.
      5. Drop rows still missing latitude/longitude/price/livable_surface
         (these can't be sensibly imputed).
      6. Re-run dedup.
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

    print(f"Cleaned shape: {df.shape} | Missing values: {df.isna().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
# 2. Build the sklearn Pipeline
#    ColumnTransformer applies different preprocessing per column group;
#    Pipeline chains preprocessing -> model into a single fit/predict object.
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FULL),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])
    return preprocessor


def build_pipeline(regressor) -> Pipeline:
    """One Pipeline object: preprocessing + model, fit and saved together."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", regressor),
    ])


REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    ),
}


# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------

def evaluate_pipeline(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

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

    return {"model": name, "train_r2": train_r2, "test_r2": test_r2,
            "mae": mae, "rmse": rmse, "mape": mape}


def train_and_evaluate_all(X_train, X_test, y_train, y_test):
    """
    Build, fit, and evaluate one full Pipeline per candidate regressor.
    Returns (results_df, fitted_pipelines) so the caller can pick a winner.
    """
    fitted_pipelines = {}
    results = []

    for name, regressor in REGRESSORS.items():
        pipeline = build_pipeline(regressor)
        pipeline.fit(X_train, y_train)          # fits preprocessor + model together
        fitted_pipelines[name] = pipeline
        results.append(evaluate_pipeline(name, pipeline, X_train, y_train, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)
    return results_df, fitted_pipelines


# ---------------------------------------------------------------------------
# 4. Save / load — ONE file now instead of four
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "pipeline.pkl")
    joblib.dump(pipeline, path)
    print(f"Saved full pipeline -> {path}")


def load_pipeline(models_dir: str = MODELS_DIR) -> Pipeline:
    return joblib.load(os.path.join(models_dir, "pipeline.pkl"))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(data_path: str, models_dir: str = "models"):
    # 1. Clean
    df = clean_data(data_path)

    # 2. Split (raw columns in, no manual impute/encode/scale needed anymore —
    #    the pipeline itself does that, fit on X_train only)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Split shapes:", X_train.shape, X_test.shape)

    # 3. Train + evaluate Linear Regression, Random Forest, XGBoost
    results_df, fitted_pipelines = train_and_evaluate_all(X_train, X_test, y_train, y_test)

    print("\n=== Model comparison (sorted by test R²) ===")
    print(results_df.to_string(index=False))

    # 4. Pick the winner and persist the ENTIRE pipeline (preprocessing + model)
    best_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (test R² = {results_df.iloc[0]['test_r2']:.4f})")

    save_pipeline(best_pipeline, models_dir=models_dir)

    return results_df, best_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Immo Eliza price models.")
    parser.add_argument("--data", default="Data/SaleCleanForAnalysis.csv", help="Path to the raw listings CSV.")
    parser.add_argument("--models-dir", default="models", help="Where to save pipeline.pkl.")
    args = parser.parse_args()

    run(args.data, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
