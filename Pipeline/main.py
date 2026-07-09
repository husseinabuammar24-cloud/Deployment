# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# define models folder
MODELS_DIR = "models"


def simplify_availability(val):
    """Must match train_model.py exactly, or the encoder sees unseen values."""
    if pd.isna(val):
        return "Unknown"
    if val in ["On contract", "Immediately", "Negotiable"]:
        return val
    if str(val).startswith("From"):
        return "Future date"
    return val


# Input schema - only columns the model actually sees
class propertyData(BaseModel):
    longitude: float
    latitude: float
    property_type: str
    property_subtype: str
    date_of_construction: float
    property_condition: str
    livable_surface: float
    number_of_bedrooms: float
    number_of_bathrooms: float
    elevator: bool
    furnished: bool
    availability: str
    province: str
    land_surface: float
    balcony: bool
    swimming_pool: bool

    class Config:
        schema_extra = {
            "example": {
                "longitude": 4.3517,
                "latitude": 50.8503,
                "property_type": "House",
                "property_subtype": "Villa",
                "date_of_construction": 1998.0,
                "property_condition": "Good",
                "livable_surface": 150.0,
                "number_of_bedrooms": 3.0,
                "number_of_bathrooms": 2.0,
                "elevator": False,
                "furnished": False,
                "availability": "Immediately",
                "province": "Brussels",
                "land_surface": 300.0,
                "balcony": True,
                "swimming_pool": False,
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="Property Price Predictor",
    description="Predicts prices from Belgium properties datasets",
    version="1.0.0",
)

# Load the ONE fitted pipeline (imputer + encoder + scaler + model, all in one).
# Replaces the old 5 separate pickle loads (model/encoder/scaler/impute_values/
# feature_names) — the Pipeline object already knows the exact preprocessing
# steps and column order it was trained with.
pipeline = joblib.load(os.path.join(MODELS_DIR, "pipeline.pkl"))


@app.post("/predict")
def predict_price(property: propertyData):
    """
    Predict a property's sale price from its listing features.
    """
    df = pd.DataFrame([property.dict()])

    # Same cleaning step training applied
    df["availability_clean"] = df["availability"].apply(simplify_availability)
    df = df.drop(columns=["availability"])

    # Binary flags -> int, matching training
    binary_cols = ["balcony", "swimming_pool", "elevator", "furnished"]
    df[binary_cols] = df[binary_cols].astype(int)

    # The pipeline handles imputation, encoding, and scaling internally,
    # in the exact order it was fit — no manual steps needed here anymore.
    prediction = pipeline.predict(df)[0]
    return {"predicted_price": round(float(prediction), 2)}


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "property_price_v1",
        "model_loaded": pipeline is not None,
    }


# Start uvicorn (bash: uvicorn app.main:app --reload) THEN -> http://127.0.0.1:8001/docs
