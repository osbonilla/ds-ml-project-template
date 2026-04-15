"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path


def fetch_housing_data(housing_url: str, housing_path: str):
    """
    Descarga y extrae el dataset de California Housing.

    1. Crea el directorio housing_path si no existe.
    2. Descarga el archivo .tgz desde housing_url con urllib.request.urlretrieve.
    3. Extrae el contenido en housing_path con tarfile.open.

    Args:
        housing_url : URL del archivo .tgz a descargar.
        housing_path: Ruta local donde se guardarán los datos.

    Resultado:
        data/raw/housing/housing.csv listo para usar.
    """
    # 1. Crear el directorio si no existe
    os.makedirs(housing_path, exist_ok=True)

    # 2. Descargar el archivo .tgz
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print(f"Descargando desde: {housing_url}")
    urllib.request.urlretrieve(housing_url, tgz_path)
    print(f"Descargado: {tgz_path}")

    # 3. Extraer el contenido del .tgz
    print(f"Extrayendo en: {housing_path}")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(path=housing_path)

    print(f"Datos disponibles en: {housing_path}/housing/housing.csv")


if __name__ == "__main__":
    URL  = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)
