"""
Script para dividir los datos en entrenamiento, validación y prueba.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    Divide el dataset en tres conjuntos y los guarda en interim_data_path.

    División:
        100% dataset original
        ├── test_set.csv   (20%)  — se guarda y NO se toca hasta evaluación final
        └── 80% restante
            ├── train_set.csv  (64% del total) — para entrenar el modelo
            └── val_set.csv    (16% del total) — para validar durante el desarrollo

    Estrategia:
        Se usa StratifiedShuffleSplit sobre income_cat (categorías de median_income)
        para garantizar que los tres conjuntos tengan la misma distribución de ingresos.
        Esto evita sesgo en la evaluación.

    Args:
        raw_data_path    : Ruta al CSV crudo (data/raw/housing/housing.csv).
        interim_data_path: Carpeta destino para los tres archivos CSV.

    ⚠️  Data Leakage:
        El split ocurre sobre datos CRUDOS, antes de cualquier transformación.
        Ninguna estadística de val_set ni test_set debe usarse en el entrenamiento.
    """
    # 1. Leer el CSV crudo
    print(f"Leyendo datos desde: {raw_data_path}")
    housing = pd.read_csv(raw_data_path)
    print(f"Dataset completo : {housing.shape[0]:,} filas x {housing.shape[1]} columnas")

    # 2. Crear variable auxiliar para estratificar por median_income
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # 3. Primera división: 80% train+val  /  20% test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_val_idx, test_idx in splitter.split(housing, housing["income_cat"]):
        train_val = housing.iloc[train_val_idx]
        test_set  = housing.iloc[test_idx]

    # 4. Segunda división sobre el 80%: 80% train  /  20% val
    #    0.2 del 80% = 16% del total original
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in splitter2.split(train_val, train_val["income_cat"]):
        train_set = train_val.iloc[train_idx]
        val_set   = train_val.iloc[val_idx]

    # 5. Eliminar la columna auxiliar income_cat de los tres conjuntos
    train_set = train_set.drop(columns=["income_cat"])
    val_set   = val_set.drop(columns=["income_cat"])
    test_set  = test_set.drop(columns=["income_cat"])

    # 6. Guardar los tres conjuntos
    Path(interim_data_path).mkdir(parents=True, exist_ok=True)

    train_path = Path(interim_data_path) / "train_set.csv"
    val_path   = Path(interim_data_path) / "val_set.csv"
    test_path  = Path(interim_data_path) / "test_set.csv"

    train_set.to_csv(train_path, index=False)
    val_set.to_csv(val_path,     index=False)
    test_set.to_csv(test_path,   index=False)

    total = len(housing)
    print(f"\nTrain guardado : {train_path}  ({len(train_set):,} filas — {len(train_set)/total:.0%})")
    print(f"Val   guardado : {val_path}    ({len(val_set):,}  filas — {len(val_set)/total:.0%})")
    print(f"Test  guardado : {test_path}   ({len(test_set):,}  filas — {len(test_set)/total:.0%})")
    print(f"\nTotal verificado : {len(train_set) + len(val_set) + len(test_set):,} filas")


if __name__ == "__main__":
    RAW_PATH     = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
