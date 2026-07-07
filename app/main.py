# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os

# define models folder and columns 
MODELS_DIR = "models"

NUMERIC_COLS = [
    "livable_surface", "number_of_bedrooms", "number_of_bathrooms",
    "land_surface", "date_of_construction",
]
CATEGORICAL_COLS = [
    "property_type", "property_subtype", "property_condition",
    "province", "availability_clean",
]
NUMERIC_FULL = NUMERIC_COLS + ["latitude", "longitude"]
BINARY_COLS = ["balcony", "swimming_pool", "elevator", "furnished"]

def simplify_availability(val):
    """Must match train_model.py exactly, or the encoder sees unseen values."""
    if pd.isna(val):
        return "Unknown"
    if val in ["On contract", "Immediately", "Negotiable"]:
        return val
    if str(val).startswith("From"):
        return "Future date"
    return val


# Input schema - only columns the model actually sees(DROP_COLSDefine removed)
class propertyData (BaseModel):
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
        '''
        Class provides a sample request body for the /docs Swagger UI —
        purely cosmetic, has no effect on validation or prediction.
        
        '''

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
                "swimming_pool": False
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="Property Price Predictor",
    description="Predicts prices from Belgium properties datasets",
    version="1.0.0"
)

# Load model + every artifact preprocess_data() fit during training
with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, "impute_values.pkl"), "rb") as f:
    impute_values = pickle.load(f)
with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)


@app.post("/predict")
def predict_price(property: propertyData):
    """
    Predict a property's sale price from its listing features.
    """

    df = pd.DataFrame([property.dict()])
    
    # Same cleaning step training applied
    df["availability_clean"] = df["availability"].apply(simplify_availability)
    df = df.drop(columns=["availability"])

    # Binary flags -> int, matching preprocess_data
    df[BINARY_COLS] = df[BINARY_COLS].astype(int)

    # Impute (fitted values from training, applied — nothing should actually
    # be missing here since Pydantic requires all fields, but kept for parity)
    for col, value in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # One-hot encode categoricals with the SAME fitted encoder
    encoded_array = encoder.transform(df[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=CATEGORICAL_COLS), encoded_df], axis=1)

    # Scale numerics with the SAME fitted scaler
    df[NUMERIC_FULL] = scaler.transform(df[NUMERIC_FULL])

    # Enforce exact training-time column order
    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(df)[0]
    return {"predicted_price": round(float(prediction), 2)}

    

@app.get("/")
def health_check():
    return {"status": "healthy", 
            "model": "property_price_v1",
            "model_loaded": model is not None,
            }


# Start uvicorn (bashuvicorn app.main:app --reload) THEN -> http://127.0.0.1:8001/docs
