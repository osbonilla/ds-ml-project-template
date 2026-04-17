"""
API de Predicción de Precios de Vivienda — California Housing
FastAPI + RandomForestRegressor (tuned)

El endpoint POST /predict recibe los datos crudos del bloque censal,
aplica el mismo pipeline de transformación del notebook 02
(imputer → encoder → feature engineering → scaler)
y retorna la predicción de median_house_value en USD.

Uso:
    uvicorn src.api.main:app --reload
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "API de Predicción de Precios de Vivienda (California)",
    description = "Predice el valor medio de vivienda (USD) a partir de datos del bloque censal.",
    version     = "1.0",
)

# ─── Rutas de artefactos ──────────────────────────────────────────────────────
MODELS_PATH = Path("models")

# ─── Variables globales ───────────────────────────────────────────────────────
model   = None
imputer = None
encoder = None
scaler  = None

# Columnas en el orden exacto con que se entrenó el modelo
# (latitud y longitud excluidas — no usadas en el modelo)
FEATURE_COLS = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

# Orden ordinal de ocean_proximity (definido empíricamente por mediana de precios)
OCEAN_ORDER = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]


# ─── Esquema de entrada ───────────────────────────────────────────────────────
class HousingFeatures(BaseModel):
    housing_median_age : float = Field(..., example=28.0,   description="Edad mediana de las viviendas del bloque")
    total_rooms        : float = Field(..., example=2500.0, description="Total de habitaciones en el bloque")
    total_bedrooms     : float = Field(..., example=500.0,  description="Total de dormitorios en el bloque")
    population         : float = Field(..., example=1200.0, description="Población total del bloque censal")
    households         : float = Field(..., example=400.0,  description="Número de hogares en el bloque")
    median_income      : float = Field(..., example=4.5,    description="Ingreso mediano en decenas de miles de USD")
    ocean_proximity    : str   = Field(..., example="<1H OCEAN",
                                       description="Proximidad al océano: INLAND | <1H OCEAN | NEAR OCEAN | NEAR BAY | ISLAND")


# ─── Carga de artefactos al iniciar ──────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, imputer, encoder, scaler
    try:
        model   = joblib.load(MODELS_PATH / "best_model.pkl")
        imputer = joblib.load(MODELS_PATH / "imputer.pkl")
        encoder = joblib.load(MODELS_PATH / "encoder.pkl")
        scaler  = joblib.load(MODELS_PATH / "scaler.pkl")
        print("Artefactos cargados correctamente.")
    except Exception as e:
        print(f"Advertencia: No se pudieron cargar los artefactos — {e}")


# ─── Archivos estáticos ───────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return FileResponse("src/api/static/index.html")


@app.get("/health")
def health():
    loaded = all(x is not None for x in [model, imputer, encoder, scaler])
    return {
        "status"    : "ok" if loaded else "error",
        "artefactos": {
            "best_model": model   is not None,
            "imputer"   : imputer is not None,
            "encoder"   : encoder is not None,
            "scaler"    : scaler  is not None,
        }
    }


@app.post("/predict")
def predict_price(features: HousingFeatures):
    if any(x is None for x in [model, imputer, encoder, scaler]):
        raise HTTPException(status_code=503, detail="Modelo o artefactos no cargados.")

    if features.ocean_proximity not in OCEAN_ORDER:
        raise HTTPException(
            status_code=422,
            detail=f"ocean_proximity inválido. Valores aceptados: {OCEAN_ORDER}"
        )

    # 1. DataFrame con datos crudos — lat/lon como placeholder requerido por el imputer
    raw = pd.DataFrame([{
        "longitude"         : 0.0,
        "latitude"          : 0.0,
        "housing_median_age": features.housing_median_age,
        "total_rooms"       : features.total_rooms,
        "total_bedrooms"    : features.total_bedrooms,
        "population"        : features.population,
        "households"        : features.households,
        "median_income"     : features.median_income,
        "ocean_proximity"   : features.ocean_proximity,
    }])

    # 2. Imputación — el imputer fue fitteado con lat/lon incluidas
    num_cols      = raw.select_dtypes(include="number").columns
    raw[num_cols] = imputer.transform(raw[num_cols])

    # 3. Codificación ordinal de ocean_proximity
    ocean_encoded          = encoder.transform(raw[["ocean_proximity"]])[0][0]
    raw["ocean_proximity"] = ocean_encoded

    # 4. Feature engineering
    raw["rooms_per_household"]      = raw["total_rooms"]    / raw["households"]
    raw["bedrooms_per_room"]        = raw["total_bedrooms"] / raw["total_rooms"]
    raw["population_per_household"] = raw["population"]     / raw["households"]

    # 5. Escalado — solo FEATURE_COLS (lat/lon excluidas)
    raw_scaled               = raw[FEATURE_COLS].copy()
    raw_scaled[FEATURE_COLS] = scaler.transform(raw_scaled[FEATURE_COLS])

    # 6. Predicción
    prediction = model.predict(raw_scaled[FEATURE_COLS])[0]

    return {
        "predicted_price"        : round(float(prediction), 2),
        "ocean_proximity_encoded": int(ocean_encoded),
    }


# uvicorn src.api.main:app --reload