# Proyecto Final — Fundamentos de Ciencia de Datos — USFQ  
Desarrollado por **Oldrin Santiago Bonilla Cáceres** 

Pipeline completo de Machine Learning para la predicción del valor mediano de vivienda en distritos de California: desde los datos crudos hasta dos versiones de API desplegadas con interfaz web.

---

## Índice

- [Descripción](#descripción)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Stack tecnológico](#stack-tecnológico)
- [Instalación](#instalación)
- [Pipeline de ejecución](#pipeline-de-ejecución)
- [Resultados del modelo](#resultados-del-modelo)
- [Despliegue — API v1](#despliegue--api-v1)
- [Despliegue — API v2 Geográfica](#despliegue--api-v2-geográfica)
- [Carpetas adicionales](#carpetas-adicionales)

---

## Descripción

El proyecto toma el dataset **California Housing** (20,640 bloques censales) y construye un modelo predictivo de `median_house_value` (USD) pasando por todas las fases de un producto de datos real:

1. Descarga automatizada y partición estratificada de datos
2. EDA profundo con visualizaciones estáticas y geoespaciales
3. Limpieza, imputación con KNNImputer y feature engineering
4. Benchmark de 4 algoritmos con fine-tuning via GridSearchCV
5. Despliegue en FastAPI con interfaz web — dos versiones

---

## Estructura del repositorio

```
ds-ml-project-template/
│
├── data/                          ← NO se sube a Git (.gitignore)
│   ├── raw/                       ← housing.tgz descargado por make_dataset.py
│   ├── interim/                   ← train_set.csv, val_set.csv, test_set.csv
│   └── processed/                 ← train_processed.csv (post pipeline)
│
├── models/                        ← NO se sube a Git
│   ├── best_model.pkl             ← RandomForestRegressor tuned
│   ├── imputer.pkl                ← KNNImputer fitteado en train
│   ├── encoder.pkl                ← OrdinalEncoder (ocean_proximity)
│   └── scaler.pkl                 ← StandardScaler fitteado en train
│
├── notebooks/
│   ├── 01_exploracion.ipynb       ← EDA completo
│   ├── 02_limpieza_enriquecimiento.ipynb  ← Feature engineering
│   └── 03_experimentacion.ipynb   ← Benchmark + fine-tuning + conclusiones
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py        ← Descarga housing.tgz
│   │   └── split_data.py          ← Partición train / val / test
│   ├── features/
│   │   └── build_features.py      ← Pipeline de transformación reproducible
│   ├── models/
│   │   └── train_model.py         ← Entrena y serializa best_model.pkl
│   └── api/
│       ├── main.py                ← API v1 — formulario estático
│       ├── main_geo.py            ← API v2 — mapa interactivo Leaflet.js
│       └── static/
│           ├── index.html         ← UI v1
│           ├── style.css
│           ├── app.js
│           └── geo/               ← UI v2 geográfica
│               ├── index.html
│               ├── style.css
│               └── app.js
│
├── reports/
│   └── figures/                   ← Gráficas generadas por los notebooks
│
├── references/                    ← Fuentes y bibliografía
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Stack tecnológico

| Categoría | Librería / Herramienta |
|-----------|------------------------|
| Datos | pandas, numpy |
| Visualización | matplotlib, seaborn, folium, contextily, geopandas |
| ML | scikit-learn (KNNImputer, OrdinalEncoder, StandardScaler, RandomForestRegressor, GridSearchCV) |
| Serialización | joblib |
| API | FastAPI, uvicorn, pydantic |
| Frontend | HTML/CSS/JS vanilla, Leaflet.js |
| Entorno | Python 3.13, venv |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/osbonilla/ds-ml-project-template.git
cd ds-ml-project-template

# 2. Crear entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Pipeline de ejecución

El proyecto se ejecuta en orden. Cada paso depende del anterior.

### Paso 1 — Descargar datos

```bash
python src/data/make_dataset.py
```

Descarga `housing.tgz` desde el repositorio de Aurélien Géron y lo extrae en `data/raw/`.

### Paso 2 — Partición de datos

```bash
python src/data/split_data.py
```

Divide el dataset en **train (64%) / val (16%) / test (20%)** con estratificación por `median_income` para evitar data leakage. Guarda los CSV en `data/interim/`.

### Paso 3 — Pipeline de features

```bash
python src/features/build_features.py
```

Aplica sobre train el pipeline completo:

| Paso | Transformación | Decisión |
|------|----------------|----------|
| Imputación | KNNImputer (k=5) sobre `total_bedrooms` | Vecinos geográficamente similares → más coherente que media global |
| Encoding | OrdinalEncoder en `ocean_proximity` | Variable ordinal: INLAND(0) → ISLAND(4) por mediana de precio |
| Feature Eng. | `rooms_per_household`, `bedrooms_per_room`, `population_per_household` | Normalización por hogar mejora correlación con target |
| Escalado | StandardScaler (excluye lat/lon) | Robusto a outliers vs MinMaxScaler |

Guarda `data/processed/train_processed.csv` y los artefactos `models/imputer.pkl`, `models/encoder.pkl`, `models/scaler.pkl`.

### Paso 4 — Entrenamiento del modelo

```bash
python src/models/train_model.py
```

Entrena `RandomForestRegressor` con los hiperparámetros óptimos del GridSearchCV sobre train, evalúa en val y test, y guarda `models/best_model.pkl`.

---

## Resultados del modelo

### Benchmark completo (notebook 03)

| Modelo | RMSE Train | RMSE Val | R² Val | Diagnóstico |
|--------|-----------|---------|--------|-------------|
| **RandomForest (tuned)** | $39,795 | $58,515 | 0.7365 | ✅ Balance correcto |
| RandomForest (base) | $22,141 | $58,781 | 0.7341 | ⚠️ Overfitting (gap $36,640) |
| DecisionTree (tuned) | $54,854 | $63,983 | 0.6850 | ✅ Balance correcto |
| SGDRegressor (tuned) | $71,003 | $70,239 | 0.6204 | ⚠️ Underfitting |
| LinearRegression (base) | $70,978 | $70,298 | 0.6197 | ⚠️ Underfitting |
| LinearRegression (Ridge) | $70,978 | $70,298 | 0.6197 | ⚠️ Underfitting |
| SGDRegressor (base) | $162,944 | $73,206 | 0.5876 | ⚠️ Underfitting severo |
| DecisionTree (base) | $0 | $80,426 | 0.5023 | ⚠️ Overfitting total |

### Modelo ganador — RandomForestRegressor (tuned)

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,   # corrige el overfitting del base
    random_state=42,
    n_jobs=-1,
)
```

**Hiperparámetros** encontrados por `GridSearchCV(cv=5, scoring='neg_mean_squared_error')`.

**Por qué RandomForest:**
- Menor RMSE Val ($58,515) de todo el benchmark
- R² = 0.7365 — explica el 73.65% de la varianza del precio
- Gap de $18,720 < umbral de $20,000 → sin overfitting
- MAE Val = $41,343 — el menor del benchmark
- `min_samples_leaf=5` redujo el gap del base ($36,640) en $17,920

---

## Despliegue — API v1

Interfaz web estática con formulario de ingreso manual de datos.

```bash
uvicorn src.api.main:app --reload --port 8000
```

Abre `http://127.0.0.1:8000`

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Interfaz web (HTML) |
| `GET` | `/health` | Estado del modelo y artefactos |
| `POST` | `/predict` | Predicción de precio |
| `GET` | `/docs` | Swagger UI automático |

### Ejemplo de request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "housing_median_age": 28,
    "total_rooms": 2500,
    "total_bedrooms": 500,
    "population": 1200,
    "households": 400,
    "median_income": 4.5,
    "ocean_proximity": "<1H OCEAN"
  }'
```

```json
{
  "predicted_price": 245800.0,
  "ocean_proximity_encoded": 1
}
```

---

## Despliegue — API v2 Geográfica

Versión extendida con mapa interactivo Leaflet. El usuario hace clic en California para seleccionar la ubicación — las coordenadas se autocompletan y alimentan al modelo con `latitude` y `longitude` reales (no placeholders), mejorando la imputación KNN.

```bash
uvicorn src.api.main_geo:app --reload --port 8001
```

Abre `http://127.0.0.1:8001/geo`

### Diferencias respecto a v1

| Aspecto | v1 | v2 |
|---------|----|----|
| Coordenadas | Placeholder `0.0` | Lat/lon reales del mapa |
| Imputación KNN | Vecinos globales | Vecinos geográficamente cercanos |
| Interfaz | Formulario | Mapa + formulario |
| Endpoint | `/predict` | `/predict/geo` |
| Puerto | 8000 | 8001 |

### Validaciones del mapa

El mapa incluye dos validaciones antes de permitir la predicción:

1. **Bounding box de California** (`32.5°–42.0° N`, `-124.5°–-114.1° W`) — fuera del rango muestra un toast de advertencia y bloquea el formulario.
2. **Detección de océano** — línea costera aproximada de 53 puntos interpolada por latitud. Si el clic cae al oeste de la costa (Pacífico), bloquea con mensaje.

### Ejemplo de request

```bash
curl -X POST http://127.0.0.1:8001/predict/geo \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 34.05,
    "longitude": -118.24,
    "housing_median_age": 28,
    "total_rooms": 2500,
    "total_bedrooms": 500,
    "population": 1200,
    "households": 400,
    "median_income": 4.5,
    "ocean_proximity": "<1H OCEAN"
  }'
```

```json
{
  "predicted_price": 251400.0,
  "latitude": 34.05,
  "longitude": -118.24,
  "ocean_proximity_encoded": 1
}
```

---

## Carpetas adicionales

### `references/`

**Dataset y Machine Learning**
- Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. O'Reilly. (fuente del dataset California Housing y referencia principal de RandomForest, KNNImputer y GridSearchCV)
- Kumar, R. — *Machine Learning Quick Reference: Quick and essential machine learning hacks for training smart data models*. Packt Publishing.

**API y Despliegue**
- Lauret, A. — *The Design of Web APIs, Second Edition*. Manning Publications. (diseño de endpoints REST, convenciones HTTP, estructura de responses)
- Clark, W. E. — *API Development Made Easy: A Practical Guide with Examples*. (referencia práctica para la implementación con FastAPI)

**Frontend**
- Godbolt, M. — *Frontend Architecture for Design Systems: A Modern Blueprint for Scalable and Sustainable Websites*. O'Reilly. (estructura de la UI, separación de responsabilidades HTML/CSS/JS)

**Geoespacial**
- Rey, S., Arribas-Bel, D. & Wolf, L. J. — *Geographic Data Science with Python*. (referencia para el mapa interactivo, validación de coordenadas y análisis geoespacial con geopandas y contextily)

---

## Variables del modelo

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `housing_median_age` | float | Edad mediana de las viviendas del bloque |
| `total_rooms` | float | Total de habitaciones en el bloque |
| `total_bedrooms` | float | Total de dormitorios (KNNImputer si NaN) |
| `population` | float | Población total del bloque censal |
| `households` | float | Número de hogares |
| `median_income` | float | Ingreso mediano (×$10,000) |
| `ocean_proximity` | str → int | INLAND=0, <1H OCEAN=1, NEAR OCEAN=2, NEAR BAY=3, ISLAND=4 |
| `rooms_per_household` | float | `total_rooms / households` (derivada) |
| `bedrooms_per_room` | float | `total_bedrooms / total_rooms` (derivada) |
| `population_per_household` | float | `population / households` (derivada) |

`latitude` y `longitude` se usan en el pipeline de imputación pero **no entran al modelo**.

---

## Autor

**Oldrin Santiago Bonilla Cáceres**  
GitHub: [@osbonilla](https://github.com/osbonilla)

Proyecto Final — Fundamentos de Ciencia de Datos · Universidad San Francisco de Quito (USFQ)