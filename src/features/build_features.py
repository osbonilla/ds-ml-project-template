"""
Script para construir las features finales a partir de los datos crudos.

Traslada la lógica de notebooks/02_limpieza_enriquecimiento.ipynb
a funciones reutilizables y reproducibles.

Pipeline:
    1. Imputación de NaN con KNNImputer (fit solo en train pero funciona para datos crudos)
    2. Codificación ordinal de ocean_proximity
    3. Feature engineering: nuevas variables combinadas
    4. Escalado con StandardScaler (fit solo en train pero funciona para datos crudos)
    5. Guardado del CSV procesado + artefactos (.pkl)

Uso:
    python src/features/build_features.py

Entrada : data/interim/train_set.csv
Salida  : data/processed/train_processed.csv
          models/imputer.pkl
          models/encoder.pkl
          models/scaler.pkl
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ─── Constantes ───────────────────────────────────────────────────────────────
INTERIM_PATH   = Path("data/interim/")
PROCESSED_PATH = Path("data/processed/")
MODELS_PATH    = Path("models/")

TARGET = "median_house_value"

# Coordenadas geográficas — no se escalan (relación espacial, no lineal)
EXCLUDE_FROM_SCALING = ["latitude", "longitude"]

# Orden empírico determinado por mediana de median_house_value por categoría
OCEAN_ORDER = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]


# ─── 1. Imputación ────────────────────────────────────────────────────────────

def impute_missing(
    X_train: pd.DataFrame,
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, KNNImputer]:
    """
    Imputa valores faltantes usando KNNImputer fitteado solo en train.

    Se usa KNN en lugar de media/mediana porque los valores globales ignoran
    el contexto del bloque censal. Bloques con características similares
    tendrán dormitorios similares — KNN captura esa coherencia local.

    Args:
        X_train     : DataFrame de features sin el target.
        n_neighbors : Vecinos para la imputación. Default: 5.

    Returns:
        (X_train_imputado, imputer)
    """
    num_cols = X_train.select_dtypes(include="number").columns
    X_train  = X_train.copy()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])

    nan_restantes = X_train.isnull().sum().sum()
    print(f"  KNNImputer(n_neighbors={n_neighbors}) — NaN restantes: {nan_restantes}")
    return X_train, imputer


# ─── 2. Codificación Ordinal ──────────────────────────────────────────────────

def encode_ocean(
    X_train: pd.DataFrame,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Aplica codificación ordinal a ocean_proximity fitteada solo en train.

    Se usa codificación ordinal (no OHE) porque existe un orden empírico claro:
    a mayor proximidad al mar, mayor mediana de median_house_value.

    Orden: INLAND(0) < <1H OCEAN(1) < NEAR OCEAN(2) < NEAR BAY(3) < ISLAND(4)

    Args:
        X_train : DataFrame con columna ocean_proximity de tipo string.

    Returns:
        (X_train_encoded, encoder)
    """
    X_train = X_train.copy()

    encoder = OrdinalEncoder(
        categories     = [OCEAN_ORDER],
        handle_unknown = "use_encoded_value",
        unknown_value  = -1,
    )

    X_train["ocean_proximity"] = encoder.fit_transform(X_train[["ocean_proximity"]])

    print("  Codificación ordinal — ocean_proximity:")
    for cat, val in zip(OCEAN_ORDER, range(len(OCEAN_ORDER))):
        print(f"    {cat:<15} → {val}")

    return X_train, encoder


# ─── 3. Feature Engineering ───────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas variables combinadas normalizadas por hogar.

    | Variable                  | Fórmula                        | Por qué es útil                          |
    |---------------------------|--------------------------------|------------------------------------------|
    | rooms_per_household       | total_rooms / households       | Tamaño real de la vivienda promedio      |
    | bedrooms_per_room         | total_bedrooms / total_rooms   | Proporción dormitorios vs total          |
    | population_per_household  | population / households        | Densidad de ocupación por hogar          |

    Args:
        df : DataFrame con columnas total_rooms, total_bedrooms,
             households y population.

    Returns:
        DataFrame con las 3 nuevas columnas añadidas.
    """
    df = df.copy()
    df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"]     / df["households"]

    print("  Nuevas features: rooms_per_household, bedrooms_per_room, population_per_household")
    return df


# ─── 4. Escalado ─────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Aplica StandardScaler a columnas numéricas fitteado solo en train.

    Excluye latitude y longitude — son coordenadas geográficas cuya
    relación con el precio es espacial, no lineal. Escalarlas no aporta.

    Se elige StandardScaler sobre MinMaxScaler porque las variables
    tienen outliers significativos (zonas de alta densidad urbana).

    Args:
        X_train : DataFrame de features ya imputadas, codificadas y enriquecidas.

    Returns:
        (X_train_escalado, scaler)
    """
    num_cols = [c for c in X_train.columns if c not in EXCLUDE_FROM_SCALING]
    X_train  = X_train.copy()

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

    print(f"  StandardScaler aplicado a ({len(num_cols)}) columnas: {num_cols}")
    print(f"  Excluidas: {EXCLUDE_FROM_SCALING}")
    return X_train, scaler


# ─── Pipeline completo ────────────────────────────────────────────────────────

def build_features(
    interim_path: Path   = INTERIM_PATH,
    processed_path: Path = PROCESSED_PATH,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering sobre train únicamente.

        1. Carga train_set.csv
        2. Separa features (X) y target (y)
        3. Imputa NaN con KNNImputer
        4. Codificación ordinal de ocean_proximity
        5. Feature engineering (nuevas variables)
        6. Escalado con StandardScaler
        7. Guarda train_processed.csv + artefactos .pkl

    Args:
        interim_path   : Directorio con train_set.csv.
        processed_path : Directorio de salida para el CSV procesado.
        save_artifacts : Si True, guarda imputer, encoder y scaler en models/.

    Returns:
        X_train procesado (sin target).
    """
    processed_path.mkdir(parents=True, exist_ok=True)

    # 1. Carga
    print("=" * 55)
    print("PIPELINE — build_features.py")
    print("=" * 55)
    print("\n[1/6] Cargando datos...")
    train   = pd.read_csv(interim_path / "train_set.csv")
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    print(f"  train_set.csv cargado: {X_train.shape[0]:,} filas x {X_train.shape[1]} columnas")
    print(f"  NaN iniciales: {X_train.isnull().sum().sum()}")

    # 2. Imputación
    print("\n[2/6] Imputando valores faltantes...")
    X_train, imputer = impute_missing(X_train)

    # 3. Codificación ordinal
    print("\n[3/6] Codificando ocean_proximity (ordinal)...")
    X_train, encoder = encode_ocean(X_train)

    # 4. Feature engineering
    print("\n[4/6] Creando nuevas features...")
    X_train = add_features(X_train)

    # 5. Escalado
    print("\n[5/6] Escalando variables numéricas...")
    X_train, scaler = scale_features(X_train)

    # 6. Guardado
    print("\n[6/6] Guardando resultados...")
    out = X_train.copy()
    out[TARGET] = y_train.values
    out.to_csv(processed_path / "train_processed.csv", index=False)
    print(f"train_processed.csv → {out.shape[0]:,} filas x {out.shape[1]} columnas")

    if save_artifacts:
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        joblib.dump(imputer, MODELS_PATH / "imputer.pkl")
        joblib.dump(encoder, MODELS_PATH / "encoder.pkl")
        joblib.dump(scaler,  MODELS_PATH / "scaler.pkl")
        print(f"imputer.pkl, encoder.pkl, scaler.pkl → {MODELS_PATH}")

    # Resumen final
    print("\n" + "=" * 55)
    print("RESUMEN FINAL")
    print("=" * 55)
    nan = X_train.isnull().sum().sum()
    obj = (X_train.dtypes == object).sum()
    print(f"  shape          : {X_train.shape}")
    print(f"  NaN            : {nan}")
    print(f"  columnas texto : {obj}")
    print(f"\n  Columnas finales ({len(X_train.columns)}):")
    for col in X_train.columns:
        print(f"    {col:<30} {str(X_train[col].dtype)}")

    return X_train


if __name__ == "__main__":
    build_features()