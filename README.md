# Demo local de MLOps

Esta primera etapa entrena de forma reproducible la red de deep learning del trabajo de compras. El objetivo es predecir el gasto promedio por cliente.

- **Airflow** orquesta una ejecucion manual del pipeline.
- **MinIO** almacena los datasets y los artefactos físicos del modelo mediante S3.
- **MLflow** registra parametros, métricas y versiones, y actúa como acceso al modelo.
- **FastAPI** consulta el modelo promovido y expone la predicción por HTTP.
- **Docker Compose** ejecuta el entorno local completo.

FastAPI carga `purchase-spend-regressor@production` desde MLflow; MLflow recupera su artefacto desde MinIO. De este modo, la API siempre usa la versión promovida por el pipeline.

## Arquitectura

`Airflow → MLflow Registry → MinIO (artefactos del modelo) → FastAPI`

Airflow ejecuta el ETL cada 30 minutos, entrena y registra cada run en MLflow. MLflow conserva la metadata del experimento y publica los artefactos del modelo en el bucket `mlflow-artifacts` de MinIO. FastAPI se conectará únicamente a MLflow mediante el alias estable `production`.

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

En Airflow, busca `train_purchase_regressor`, habilitalo si fuera necesario y ejecutalo con **Trigger DAG**. Luego se ejecuta cada 30 minutos, hora de Argentina. El DAG crea el bucket `mlops-data` y ejecuta estas etapas:

1. `upload_raw_dataset`: carga `dataset_compras.csv` en `raw/dataset_compras.csv`.
2. `eda_validate`: valida columnas, nulos, compras inválidas y consistencia demográfica; guarda el informe en `reports/eda_dataset_compras.json`.
3. `prepare_curated_dataset`: limpia transacciones, agrega por cliente y genera features sin leakage; guarda `curated/tabla_clientes_modelado.csv`.
4. `train_evaluate_and_register`: descarga exclusivamente la tabla curada desde MinIO, entrena la MLP y registra la ejecución en MLflow. Si `r2_test` es mayor o igual que el de `production`, promueve la versión con los alias `candidate` y `production`; de lo contrario, conserva la versión en producción.

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

Airflow solo instala `virtualenv`. Cada tarea crea un entorno virtual temporal: la carga usa `boto3`, EDA y preparación usan `boto3` y `pandas`, y el entrenamiento instala PyTorch, MLflow y scikit-learn. Esto evita mezclar las librerías de ML con las de Airflow. La primera ejecución demora más porque descarga esas dependencias.

## Detener

```bash
docker-compose down
```

Los datos persisten en volumenes Docker. Para eliminarlos intencionalmente: `docker-compose down --volumes`.
