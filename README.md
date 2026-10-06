# Pipeline local de MLOps para prediccion de gasto

Este proyecto implementa un pipeline de MLOps de punta a punta para estimar el gasto promedio de un cliente a partir de su historial de compras. El objetivo no es solamente entrenar una red neuronal, sino automatizar la ingesta, validacion, preparacion de datos, reentrenamiento, registro, promocion y servicio del modelo.

## Problema y dataset

El problema es una regresion supervisada. A partir de transacciones identificadas por usuario y producto, el pipeline construye una tabla agregada por cliente y predice `Gasto_Promedio`.

El dataset crudo contiene, entre otras, las columnas `User_ID`, `Product_ID`, `Age`, `Gender`, `Marital_Status`, `City_Category`, `Stay_In_Current_City_Years`, `Product_Category` y `Purchase`. El CSV no se versiona dentro del repositorio.

- **Nombre esperado:** `dataset_compras.csv`.
- **Origen de esta demo:** [archivo publico en Google Drive](https://drive.google.com/file/d/1e_FRepCpHBWWGj3TYG4HY7WHALCyR4wQ/view).
- **Configuracion:** variable `DATASET_SOURCE_URL` en `docker-compose.yml`.
- **Ubicacion de trabajo:** `s3://mlops-data/raw/dataset_compras.csv` en MinIO.

No es necesario descargar el CSV manualmente: la primera tarea del DAG usa `gdown`, obtiene el archivo desde Google Drive y lo deja en MinIO. El enlace debe continuar compartido para cualquier persona que tenga acceso a la URL. Para inspeccionarlo fuera del pipeline, se puede descargar con:

```bash
python -m pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/1e_FRepCpHBWWGj3TYG4HY7WHALCyR4wQ/view" \
  -O dataset_compras.csv
```

## Arquitectura

Flujo autocontenido de la demo:

```text
Google Drive
    -> adaptador de ingesta en Airflow
    -> MinIO: raw/dataset_compras.csv
    -> EDA y dataset curado
    -> entrenamiento PyTorch
    -> MLflow Registry
    -> FastAPI
```

El contrato de entrada del pipeline de modelado es el objeto `s3://mlops-data/raw/dataset_compras.csv`. MinIO simula el servicio S3 que se utilizaria en un entorno real.

En un escenario productivo, otro proceso ETL, independiente de este DAG, seria responsable de actualizar ese objeto con informacion nueva. El DAG de entrenamiento solo consultaria MinIO, detectaria si el contenido cambio y decidiria si corresponde reentrenar. Para que la demo pueda ejecutarse sin ese sistema externo, la tarea `download_raw_dataset` representa temporalmente esa ingesta previa descargando el archivo desde Google Drive.

Airflow agenda el DAG cada 30 minutos para emular un cron productivo y permitir demostrar varias corridas en poco tiempo. Esa frecuencia es ilustrativa: segun la disponibilidad de datos y el costo de entrenamiento, en produccion podria ejecutarse una vez por dia o con otra periodicidad.

## Componentes

- **Airflow:** agenda y orquesta la ingesta, validacion, preparacion y entrenamiento.
- **MinIO:** simula S3 y almacena datasets, estado de procesamiento y artefactos.
- **MLflow:** registra parametros, metricas, artefactos y versiones del modelo.
- **PyTorch:** entrena la red neuronal de regresion.
- **FastAPI:** sirve la version promovida mediante una API REST.
- **Docker Compose:** levanta el entorno local completo.

FastAPI carga `purchase-spend-regressor@production` desde MLflow. MLflow recupera el artefacto fisico desde el bucket `mlflow-artifacts` de MinIO, por lo que la API siempre resuelve la version promovida por el pipeline.

## Estructura

```text
machine_learning_operations/
├── airflow/              # Imagen, DAG y dependencias de Airflow
├── api/                  # API REST de inferencia
├── mlflow/               # Imagen del servidor de tracking
├── docker-compose.yml    # Servicios e infraestructura local
├── CHANGELOG.md          # Cambios por entrega
└── README.md
```

## Requisitos

- Docker Desktop, Docker Engine con Compose o Colima.
- Al menos 4 GB de memoria disponibles para Docker.
- Conexion a Internet durante la construccion y la descarga del dataset.
- Puertos locales `5000`, `8000`, `8081`, `9100` y `9101` disponibles.

## Ejecucion local

Desde una terminal:

```bash
git clone https://github.com/waltergsotelo/machine_learning_operations.git
cd machine_learning_operations
docker compose up --build airflow-init
docker compose up --build -d \
  postgres minio minio-init mlflow airflow-webserver airflow-scheduler
```

Si la instalacion utiliza el binario clasico, reemplazar `docker compose` por `docker-compose`. Si el repositorio ya esta clonado, ejecutar los comandos desde su raiz, donde se encuentra `docker-compose.yml`.

Servicios disponibles:

- Airflow: http://localhost:8081 (`airflow` / `airflow`).
- MLflow: http://localhost:5000.
- MinIO Console: http://localhost:9101 (`minioadmin` / `minioadmin`).

En Airflow, buscar `train_purchase_regressor`, habilitarlo y ejecutarlo con **Trigger DAG**. La primera corrida descarga y procesa el dataset. Las siguientes comparan su contenido antes de ejecutar las etapas costosas.

Cuando la primera corrida termine correctamente y MLflow ya tenga el alias `production`, iniciar la API:

```bash
docker compose up --build -d api
```

FastAPI quedara disponible en http://localhost:8000/docs. Este orden es necesario porque la API carga el modelo promovido durante su inicio.

## Etapas del DAG

1. `download_raw_dataset`: descarga el origen configurado, calcula su SHA-256 y lo compara con el ultimo dataset procesado. Si cambio, lo publica en `raw/dataset_compras.csv`.
2. `continue_if_dataset_changed`: deja las tareas restantes en estado `skipped` cuando el hash no cambio.
3. `eda_validate`: valida columnas, filas, nulos, compras invalidas y consistencia demografica; guarda `reports/eda_dataset_compras.json`.
4. `prepare_curated_dataset`: limpia y agrega las transacciones por cliente; guarda `curated/tabla_clientes_modelado.csv`.
5. `train_evaluate_and_register`: entrena la MLP y registra la corrida en MLflow. Promueve la version cuando su `r2_test` es mayor o igual al modelo `production` vigente.
6. `mark_dataset_processed`: despues de un entrenamiento exitoso, guarda el hash y los datos de la corrida en `state/dataset_compras.json`.

El estado no se actualiza si falla la validacion, la preparacion o el entrenamiento. En ese caso, la siguiente corrida vuelve a intentar el mismo dataset. El DAG permite una sola corrida activa para evitar entrenamientos simultaneos sobre el mismo cambio.

La red utiliza tres capas ocultas de 32 neuronas, activacion ELU, dropout de `0.2`, Adam, weight decay y early stopping. MLflow registra `r2_test`, `rmse_test`, `previous_production_r2_test` e `is_promoted`.

## API de prediccion

La API resuelve el alias `production` al iniciar. `GET /model-info` informa las columnas requeridas, `GET /health` muestra la version cargada y `POST /predict` recibe las features por nombre.

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

La API responde `422` si faltan features o se envian nombres desconocidos. Hasta que exista una primera version promovida en MLflow, la API no podra cargar el alias `production`.

## Estado del proyecto

### Implementado

- Entorno reproducible con Docker Compose.
- Descarga automatica del dataset de referencia y almacenamiento en MinIO.
- Validacion, limpieza y construccion de features por cliente.
- Deteccion de cambios por SHA-256 para evitar reentrenamientos identicos.
- Entrenamiento aislado de la MLP, evaluacion y registro en MLflow.
- Promocion automatica basada en `r2_test` y servicio mediante FastAPI.

### Pendiente para un entorno productivo

- Separar la ingesta en un ETL externo que escriba directamente en S3 o MinIO.
- Adaptar el DAG de entrenamiento para consumir exclusivamente el objeto dejado por ese ETL.
- Reemplazar credenciales locales por secretos administrados.
- Incorporar pruebas automatizadas, CI/CD, monitoreo, alertas y deteccion de drift.
- Definir politicas productivas de retencion, rollback y aprobacion del modelo.

El detalle de cambios por entrega se mantiene en [CHANGELOG.md](./CHANGELOG.md).

## Integrantes

- **Alumno 1:** Juan Pablo Castelino.
- **Alumno 2:** Walter Gabriel Sotelo.
- **Alumno 3:** Esteban Leonardo Torres Martin.

## Detener el entorno

```bash
docker compose down
```

Los datos persisten en volumenes Docker. Para eliminarlos intencionalmente:

```bash
docker compose down --volumes
```
