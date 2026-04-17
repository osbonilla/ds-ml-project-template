"""
API Geográfica de Predicción de Precios de Vivienda — California Housing v2
FastAPI + RandomForestRegressor (tuned) + coordenadas reales

Diferencias respecto a main.py (v1):
    - Acepta latitude y longitude reales del usuario (clic en mapa Leaflet)
    - lat/lon se pasan al KNNImputer como valores reales — mejora la imputación
      porque busca vecinos geográficamente cercanos, no placeholders 0.0
    - Sirve la UI geográfica desde src/api/static/geo/
    - Corre en puerto 8001 para coexistir con v1 en puerto 8000
    - Endpoint /predict/geo retorna además las coordenadas del punto

Uso:
    uvicorn src.api.main_geo:app --reload --port 8001

Entrada : data/interim/  (no necesaria en runtime)
          models/best_model.pkl, imputer.pkl, encoder.pkl, scaler.pkl
Salida  : JSON con predicted_price + coordenadas
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "API Geográfica — Predicción de Precios de Vivienda (California)",
    description = "v2: igual que v1 pero con coordenadas reales y mapa interactivo.",
    version     = "2.0",
)

# ─── Rutas ────────────────────────────────────────────────────────────────────
MODELS_PATH    = Path("models")
STATIC_GEO     = Path("src/api/static/geo")

# ─── Variables globales ───────────────────────────────────────────────────────
model   = None
imputer = None
encoder = None
scaler  = None

# Columnas en el orden exacto con que se entrenó el modelo
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

# Columnas que el imputer espera (fue fitteado con lat/lon incluidas)
IMPUTER_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

OCEAN_ORDER = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# Bounding box de California — validación de coordenadas
LAT_MIN, LAT_MAX =  32.5,  42.0
LON_MIN, LON_MAX = -124.5, -114.1


# ─── Esquema de entrada ───────────────────────────────────────────────────────
class GeoHousingFeatures(BaseModel):
    """
    Igual que HousingFeatures (v1) pero con latitude y longitude reales.
    Las coordenadas provienen del clic del usuario en el mapa Leaflet.
    """
    latitude           : float = Field(..., example=34.05,
                                        description="Latitud del bloque censal (California: 32.5 – 42.0)")
    longitude          : float = Field(..., example=-118.24,
                                        description="Longitud del bloque censal (California: -124.5 – -114.1)")
    housing_median_age : float = Field(..., example=28.0,   description="Edad mediana de las viviendas")
    total_rooms        : float = Field(..., example=2500.0, description="Total de habitaciones en el bloque")
    total_bedrooms     : float = Field(..., example=500.0,  description="Total de dormitorios en el bloque")
    population         : float = Field(..., example=1200.0, description="Población total del bloque censal")
    households         : float = Field(..., example=400.0,  description="Número de hogares en el bloque")
    median_income      : float = Field(..., example=4.5,    description="Ingreso mediano en decenas de miles de USD")
    ocean_proximity    : str   = Field(..., example="<1H OCEAN",
                                        description="Proximidad al océano: INLAND | <1H OCEAN | NEAR OCEAN | NEAR BAY | ISLAND")

    @field_validator("latitude")
    @classmethod
    def validate_lat(cls, v):
        if not (LAT_MIN <= v <= LAT_MAX):
            raise ValueError(f"Latitud fuera del rango de California ({LAT_MIN} – {LAT_MAX})")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_lon(cls, v):
        if not (LON_MIN <= v <= LON_MAX):
            raise ValueError(f"Longitud fuera del rango de California ({LON_MIN} – {LON_MAX})")
        return v


# ─── Carga de artefactos ──────────────────────────────────────────────────────
@app.on_event("startup")
def load_artifacts():
    global model, imputer, encoder, scaler
    try:
        model   = joblib.load(MODELS_PATH / "best_model.pkl")
        imputer = joblib.load(MODELS_PATH / "imputer.pkl")
        encoder = joblib.load(MODELS_PATH / "encoder.pkl")
        scaler  = joblib.load(MODELS_PATH / "scaler.pkl")
        print("Artefactos cargados correctamente (v2 geo).")
    except Exception as e:
        print(f"Advertencia: No se pudieron cargar los artefactos — {e}")


# ─── Archivos estáticos ───────────────────────────────────────────────────────
app.mount("/geo/static", StaticFiles(directory=str(STATIC_GEO)), name="geo-static")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/geo")
def geo_home():
    """Sirve la interfaz geográfica con mapa Leaflet."""
    return FileResponse(str(STATIC_GEO / "index.html"))


@app.get("/health")
def health():
    """Verifica que el modelo y los artefactos estén cargados."""
    loaded = all(x is not None for x in [model, imputer, encoder, scaler])
    return {
        "status"    : "ok" if loaded else "error",
        "version"   : "2.0-geo",
        "artefactos": {
            "best_model": model   is not None,
            "imputer"   : imputer is not None,
            "encoder"   : encoder is not None,
            "scaler"    : scaler  is not None,
        },
    }


@app.post("/predict/geo")
def predict_geo(features: GeoHousingFeatures):
    """
    Predice el valor mediano de vivienda con coordenadas reales.

    Mejora sobre v1:
        El KNNImputer recibe lat/lon reales → busca vecinos geográficamente
        cercanos para imputar total_bedrooms, lo que es más preciso que
        pasar 0.0 como placeholder.

    Pipeline:
        1. KNNImputer  — con lat/lon reales (orden IMPUTER_COLS)
        2. OrdinalEncoder — ocean_proximity
        3. Feature engineering
        4. StandardScaler — solo FEATURE_COLS (lat/lon excluidas del modelo)
        5. Predicción

    Retorna:
        predicted_price    : estimación en USD
        latitude/longitude : coordenadas recibidas (para confirmar en el mapa)
        ocean_proximity_encoded : valor ordinal asignado
    """
    if any(x is None for x in [model, imputer, encoder, scaler]):
        raise HTTPException(status_code=503, detail="Modelo o artefactos no cargados.")

    if features.ocean_proximity not in OCEAN_ORDER:
        raise HTTPException(
            status_code=422,
            detail=f"ocean_proximity inválido. Valores aceptados: {OCEAN_ORDER}"
        )

    # 1. DataFrame numérico en el orden que el imputer espera
    raw = pd.DataFrame([{
        "longitude"         : features.longitude,
        "latitude"          : features.latitude,
        "housing_median_age": features.housing_median_age,
        "total_rooms"       : features.total_rooms,
        "total_bedrooms"    : features.total_bedrooms,
        "population"        : features.population,
        "households"        : features.households,
        "median_income"     : features.median_income,
        "ocean_proximity"   : features.ocean_proximity,
    }])

    # 2. Imputación con lat/lon reales
    raw[IMPUTER_COLS] = imputer.transform(raw[IMPUTER_COLS])

    # 3. Codificación ordinal
    ocean_encoded          = encoder.transform(raw[["ocean_proximity"]])[0][0]
    raw["ocean_proximity"] = ocean_encoded

    # 4. Feature engineering
    raw["rooms_per_household"]      = raw["total_rooms"]    / raw["households"]
    raw["bedrooms_per_room"]        = raw["total_bedrooms"] / raw["total_rooms"]
    raw["population_per_household"] = raw["population"]     / raw["households"]

    # 5. Escalado — solo FEATURE_COLS (lat/lon excluidas del modelo)
    raw_scaled               = raw[FEATURE_COLS].copy()
    raw_scaled[FEATURE_COLS] = scaler.transform(raw_scaled[FEATURE_COLS])

    # 6. Predicción
    prediction = model.predict(raw_scaled[FEATURE_COLS])[0]

    return {
        "predicted_price"        : round(float(prediction), 2),
        "latitude"               : features.latitude,
        "longitude"              : features.longitude,
        "ocean_proximity_encoded": int(ocean_encoded),
    }


# ─── Ejecución local ──────────────────────────────────────────────────────────
# uvicorn src.api.main_geo:app --reload --port 8001