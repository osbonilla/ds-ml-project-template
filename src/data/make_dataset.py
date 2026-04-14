"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def fetch_housing_data(housing_url: str, housing_path: str):
    """
    INSTRUCCIONES:
    1. Asegúrate de que el directorio `housing_path` exista (usa os.makedirs o Path.mkdir).
    2. Usa urllib.request.urlretrieve para descargar el archivo .tgz desde `housing_url`.
    3. Usa tarfile.open para extraer el contenido en `housing_path`.
    
    URL de los datos: "7"
    Ruta de destino recomendada: "data/raw/"
    """
    pass

if __name__ == "__main__":
    # URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    # PATH = "data/raw/"
    # fetch_housing_data(URL, PATH)
    print("Script para descargar datos... (Falta el código!)")
