"""
Script para construir las features finales a partir de los datos crudos.

Traslada la lógica de notebooks/02_limpieza_enriquecimiento.ipynb
a funciones reutilizables y reproducibles.

Uso:
    python src/features/build_features.py

Entrada  : data/interim/train_set.csv, val_set.csv, test_set.csv
Salida   : data/processed/train_processed.csv, val_processed.csv, test_processed.csv
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ─── Constantes ───────────────────────────────────────────────────────────────
INTERIM_PATH   = Path("data/interim/")
PROCESSED_PATH = Path("data/processed/")
MODELS_PATH    = Path("models/")

TARGET = "median_house_value"

# latitude y longitude son coordenadas geográficas — no se escalan
EXCLUDE_FROM_SCALING = ["latitude", "longitude"]


# ─── 1. Imputación ────────────────────────────────────────────────────────────

def impute_missing(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    n_neighbors: int = 5,
) -> tuple:
    """
    Imputa valores faltantes usando KNNImputer.

    No se usan medidas de tendencia central (media, mediana) porque son
    valores globales que ignoran el contexto del bloque censal.
    KNNImputer usa los k vecinos más similares para estimar el valor faltante,
    lo que es coherente: bloques con características parecidas tendrán
    dormitorios similares.

    El imputer se fittea SOLO sobre X_train para evitar data leakage.

    Args:
        X_train, X_val, X_test: DataFrames de features sin el target.
        n_neighbors: Número de vecinos para la imputación. Default: 5.

    Returns:
        Tupla (X_train, X_val, X_test, imputer).
    """
    num_cols = X_train.select_dtypes(include="number").columns

    imputer = KNNImputer(n_neighbors=n_neighbors)

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_val[num_cols]   = imputer.transform(X_val[num_cols])
    X_test[num_cols]  = imputer.transform(X_test[num_cols])

    print(f"  KNNImputer(n_neighbors={n_neighbors}) — NaN restantes: {X_train.isnull().sum().sum()}")
    return X_train, X_val, X_test, imputer


# ─── 2. Nuevas features ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea nuevas variables combinadas normalizadas por hogar.

    - rooms_per_household      : total_rooms / households
    - bedrooms_per_room        : total_bedrooms / total_rooms
    - population_per_household : population / households
    """
    df = df.copy()
    df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"]     / df["households"]
    return df


# ─── 3. Encoding ─────────────────────────────────────────────────────────────

def encode_ocean(
    df: pd.DataFrame,
    ohe_cols: list | None = None,
) -> tuple:
    """
    Aplica One-Hot Encoding a ocean_proximity.

    Se usa OHE (nominal) porque no existe orden natural entre las categorías.
    """
    df = df.copy()
    dummies = pd.get_dummies(df["ocean_proximity"], prefix="ocean", drop_first=True)

    if ohe_cols is None:
        ohe_cols = dummies.columns.tolist()
    else:
        dummies = dummies.reindex(columns=ohe_cols, fill_value=0)

    df = df.drop(columns=["ocean_proximity"])
    df = pd.concat([df, dummies], axis=1)
    return df, ohe_cols


# ─── 4. Escalado ─────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    ohe_cols: list,
) -> tuple:
    """
    Aplica StandardScaler a las columnas numéricas.

    Excluye latitude y longitude (coordenadas geográficas) y columnas OHE.
    El scaler se fittea SOLO en X_train.
    """
    exclude  = EXCLUDE_FROM_SCALING + ohe_cols
    num_cols = [c for c in X_train.columns if c not in exclude]

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"  Escaladas ({len(num_cols)}): {num_cols}")
    print(f"  Excluidas: {exclude}")
    return X_train, X_val, X_test, scaler


# ─── Pipeline completo ────────────────────────────────────────────────────────

def build_features(
    interim_path: Path = INTERIM_PATH,
    processed_path: Path = PROCESSED_PATH,
    save_artifacts: bool = True,
) -> tuple:
    """
    Pipeline completo de feature engineering:
        1. Carga train_set, val_set y test_set
        2. Imputa con KNNImputer (fitteado en train)
        3. Crea nuevas features combinadas
        4. One-Hot Encoding de ocean_proximity
        5. StandardScaler (excluye lat, lon y OHE)
        6. Guarda CSV procesados, imputer y scaler
    """
    processed_path.mkdir(parents=True, exist_ok=True)

    print("Cargando datos...")
    train = pd.read_csv(interim_path / "train_set.csv")
    val   = pd.read_csv(interim_path / "val_set.csv")
    test  = pd.read_csv(interim_path / "test_set.csv")

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_val,   y_val   = val.drop(columns=[TARGET]),   val[TARGET]
    X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]

    print("Imputando con KNNImputer...")
    X_train, X_val, X_test, imputer = impute_missing(X_train, X_val, X_test)

    print("Creando nuevas features...")
    X_train = add_features(X_train)
    X_val   = add_features(X_val)
    X_test  = add_features(X_test)

    print("Codificando ocean_proximity...")
    X_train, ohe_cols = encode_ocean(X_train)
    X_val,   _        = encode_ocean(X_val,  ohe_cols)
    X_test,  _        = encode_ocean(X_test, ohe_cols)

    print("Escalando variables numéricas...")
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, ohe_cols)

    for X, y, name in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        out = X.copy()
        out[TARGET] = y.values
        out.to_csv(processed_path / f"{name}_processed.csv", index=False)
        print(f"✅ {name}_processed.csv  ({out.shape[0]:,} filas x {out.shape[1]} columnas)")

    if save_artifacts:
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        joblib.dump(imputer, MODELS_PATH / "imputer.pkl")
        joblib.dump(scaler,  MODELS_PATH / "scaler.pkl")
        print(f"✅ imputer.pkl y scaler.pkl guardados en {MODELS_PATH}")

    return X_train, X_val, X_test


if __name__ == "__main__":
    build_features()
