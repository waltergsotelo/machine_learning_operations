# Demo local de MLOps

Esta primera etapa entrena de forma reproducible la red de deep learning del trabajo de compras. El objetivo es predecir el gasto promedio por cliente.

## Integrantes

- **Alumno 1:** Juan Pablo Castelino
- **Alumno 2:** Walter Gabriel Sotelo
- **Alumno 3:** Esteban Leonardo Torres Martín

- **Airflow** orquesta la ingesta diaria del dataset y el reentrenamiento que esta dispara.
- **MinIO** almacena los datasets y los artefactos físicos del modelo mediante S3.
- **MLflow** registra parametros, métricas y versiones, y actúa como acceso al modelo.
- **FastAPI** consulta el modelo promovido y expone la predicción por HTTP.
- **Docker Compose** ejecuta el entorno local completo.

FastAPI carga `purchase-spend-regressor@production` desde MLflow; MLflow recupera su artefacto desde MinIO. De este modo, la API siempre usa la versión promovida por el pipeline.

## Origen del dataset y disparo del reentrenamiento

El dataset `dataset_compras.csv` (una fila por compra: `User_ID`, `Product_ID`, `Age`, `Gender`, `Marital_Status`, `City_Category`, `Stay_In_Current_City_Years`, `Product_Category`, `Purchase`, entre otras) lo publica el equipo de Data Mining/BI en una ruta remota. En esta demo esa ruta se simula con un [archivo público en Google Drive](https://drive.google.com/file/d/1e_FRepCpHBWWGj3TYG4HY7WHALCyR4wQ/view), configurado en la variable `DATASET_SOURCE_URL` de `docker-compose.yml`. No hace falta descargarlo a mano: el pipeline lo obtiene solo.

El flujo se divide en dos DAGs conectados por un [Dataset de Airflow](https://airflow.apache.org/docs/apache-airflow/2.10.5/authoring-and-scheduling/datasets.html) (`s3://mlops-data/raw/dataset_compras.csv`):

```text
ingest_purchase_dataset (diaria)                       train_purchase_regressor
download_and_publish → continue_if_changed → announce_new_dataset ──▶ eda_validate → prepare_curated_dataset → train_evaluate_and_register
```

1. `ingest_purchase_dataset` corre una vez por día. `download_and_publish` descarga el archivo, calcula su SHA-256 y lo compara con el hash guardado como metadata del objeto ya publicado en MinIO. Solo si cambió (o si todavía no hay nada publicado) lo sube a `raw/dataset_compras.csv`.
2. Si no hubo cambios, `continue_if_changed` deja `announce_new_dataset` en `skipped` y no se avisa nada.
3. Si hubo cambios, `announce_new_dataset` emite el evento del Dataset y Airflow dispara `train_purchase_regressor`.

Es decir: **el reentrenamiento se dispara cuando llega una versión nueva del dataset**, no por calendario. También puede ejecutarse a mano con **Trigger DAG**. Como el evento se consume al disparar el entrenamiento, si este falla tras su reintento automático hay que relanzarlo a mano: la ingesta no vuelve a avisar hasta que el archivo cambie.

## Arquitectura

`Airflow → MLflow Registry → MinIO (artefactos del modelo) → FastAPI`

Airflow publica el dataset en MinIO, entrena y registra cada run en MLflow. MLflow conserva la metadata del experimento y publica los artefactos del modelo en el bucket `mlflow-artifacts` de MinIO. FastAPI se conectará únicamente a MLflow mediante el alias estable `production`.

## Estructura

```text
demo-mlops/
├── airflow/       # Imagen propia, DAGs y dependencias del orquestador
├── api/           # API REST de inferencia con FastAPI
├── mlflow/        # Imagen del servidor de tracking
└── docker-compose.yml
```

## Ejecutar

Requisito: Docker Desktop con al menos 4 GB de memoria asignada.

```bash
cd "Operaciones de machine learning/demo-mlops"
docker-compose up airflow-init
docker-compose up --build -d
```

Usa `docker compose` en lugar de `docker-compose` si tu instalacion ofrece el subcomando moderno.

- Airflow: http://localhost:8081 (`airflow` / `airflow`)
- MLflow: http://localhost:5000
- MinIO Console: http://localhost:9101 (`minioadmin` / `minioadmin`)
- API REST: http://localhost:8000/docs

En Airflow, habilita los dos DAGs, `ingest_purchase_dataset` y `train_purchase_regressor`. Al habilitar la ingesta, Airflow ejecuta su corrida más reciente: en la primera instalación crea el bucket `mlops-data`, publica el dataset y dispara el entrenamiento. Después se ejecuta todos los días a medianoche, hora de Argentina. `train_purchase_regressor` ejecuta estas etapas:

1. `eda_validate`: valida columnas, nulos, compras inválidas y consistencia demográfica; guarda el informe en `reports/eda_dataset_compras.json`.
2. `prepare_curated_dataset`: limpia transacciones, agrega por cliente y genera features sin leakage; guarda `curated/tabla_clientes_modelado.csv`.
3. `train_evaluate_and_register`: descarga exclusivamente la tabla curada desde MinIO, entrena la MLP y registra la ejecución en MLflow. Si `r2_test` es mayor o igual que el de `production`, promueve la versión con los alias `candidate` y `production`; de lo contrario, conserva la versión en producción.

La red replica la arquitectura final del TP: tres capas ocultas de 32 neuronas, activación ELU, dropout de 0.2, Adam, weight decay y early stopping. Al finalizar, MLflow muestra el experimento `purchase-spend-regression`, las métricas `r2_test`, `rmse_test`, `previous_production_r2_test` e `is_promoted`. El archivo físico del modelo queda en `s3://mlflow-artifacts`.

## API de predicción

La API resuelve el alias `production` al iniciar. Envía las features por nombre a `POST /predict`; la API las ordena internamente según el esquema del modelo:

```bash
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "Gender": 1, "Marital_Status": 0,
      "prop_cat_1": 0.4, "prop_cat_2": 0, "prop_cat_3": 0, "prop_cat_4": 0,
      "prop_cat_5": 0.6, "prop_cat_6": 0, "prop_cat_7": 0, "prop_cat_8": 0,
      "prop_cat_9": 0, "prop_cat_10": 0, "prop_cat_11": 0, "prop_cat_12": 0,
      "prop_cat_13": 0, "prop_cat_14": 0, "prop_cat_15": 0, "prop_cat_16": 0,
      "prop_cat_17": 0, "prop_cat_18": 0, "prop_cat_19": 0, "prop_cat_20": 0,
      "n_transacciones": 10, "n_categorias_distintas": 2,
      "n_productos_distintos": 8, "concentracion_hhi": 0.52,
      "tasa_recompra": 0.2, "categorias_por_compra": 0.2, "City_nivel": 2,
      "edad_18-25": 0, "edad_26-35": 1, "edad_36-45": 0, "edad_46-50": 0,
      "edad_51-55": 0, "edad_55+": 0,
      "antig_2": 0, "antig_3": 1, "antig_4": 0, "antig_5+": 0
    }
  }'
```

`GET /model-info` expone las columnas requeridas. La API responde `422` si faltan features o se envían nombres desconocidos. `GET /health` indica la versión de modelo cargada.

## Dependencias aisladas

Airflow solo instala `virtualenv`. Cada tarea corre en su propio entorno virtual: la ingesta usa `boto3` y `gdown`, EDA y preparación usan `boto3` y `pandas`, y el entrenamiento instala PyTorch, MLflow y scikit-learn. Esto evita mezclar las librerías de ML con las de Airflow.

Los entornos se guardan en el volumen `airflow-venv-cache` (`venv_cache_path` en los DAGs). Airflow crea cada uno la primera vez que lo necesita, identificado por un hash de sus requisitos, y lo reutiliza en las corridas siguientes, incluso si se recrean los contenedores. Por eso solo la primera ejecución demora más. Si cambian los requisitos de una tarea, Airflow crea un entorno nuevo automáticamente. Para forzar una reinstalación completa, con el stack detenido: `docker volume rm <proyecto>_airflow-venv-cache` (el prefijo es el nombre de la carpeta del repo, por defecto `machine_learning_operations`).

## Detener

```bash
docker-compose down
```

Los datos persisten en volumenes Docker. Para eliminarlos intencionalmente: `docker-compose down --volumes`.
