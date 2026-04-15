"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    Divide el dataset en train y test y guarda los resultados.

    1. Lee el CSV desde raw_data_path.
    2. Crea una variable auxiliar income_cat para estratificar por median_income,
       asegurando que train y test tengan la misma distribución de ingresos.
    3. Aplica StratifiedShuffleSplit sobre income_cat (evita sesgo en el split).
    4. Elimina income_cat de los conjuntos finales.
    5. Guarda train_set.csv y test_set.csv en interim_data_path.

    Args:
        raw_data_path    : Ruta al CSV crudo (data/raw/housing/housing.csv).
        interim_data_path: Carpeta destino para train_set.csv y test_set.csv.

    ⚠️  Data Leakage:
        El split ocurre sobre los datos CRUDOS, antes de cualquier
        transformación (scaling, imputación, encoding).
        Ninguna estadística del test set debe usarse durante el entrenamiento.
    """
    # 1. Leer el CSV
    print(f"Leyendo datos desde: {raw_data_path}")
    housing = pd.read_csv(raw_data_path)
    print(f"Dataset completo: {housing.shape[0]:,} filas x {housing.shape[1]} columnas")

    # 2. Crear variable auxiliar income_cat para estratificar
    #    Dividimos median_income en 5 categorías representativas.
    #    np.ceil redondea hacia arriba; clip(upper=5) agrupa los valores altos en una sola categoría.
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # 3. StratifiedShuffleSplit sobre income_cat
    #    Garantiza que la proporción de cada categoría sea igual en train y test.
    #    Esto evita que, por azar, el test tenga muchos más vecindarios ricos o pobres.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        train_set = housing.iloc[train_index]
        test_set  = housing.iloc[test_index]

    # 4. Eliminar la columna auxiliar income_cat de ambos conjuntos
    train_set = train_set.drop(columns=["income_cat"])
    test_set  = test_set.drop(columns=["income_cat"])

    # 5. Guardar los conjuntos en interim_data_path
    Path(interim_data_path).mkdir(parents=True, exist_ok=True)

    train_path = Path(interim_data_path) / "train_set.csv"
    test_path  = Path(interim_data_path) / "test_set.csv"

    train_set.to_csv(train_path, index=False)
    test_set.to_csv(test_path,  index=False)

    print(f"Train guardado: {train_path}  ({len(train_set):,} filas)")
    print(f"Test  guardado: {test_path}   ({len(test_set):,} filas)")
    print(f"\nProporción: 80% train / 20% test  |  random_state=42")


if __name__ == "__main__":
    RAW_PATH     = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
