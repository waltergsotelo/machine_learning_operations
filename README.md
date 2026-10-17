# Demo local de MLOps

Esta primera etapa entrena de forma reproducible la red de deep learning del trabajo de compras. El objetivo es predecir el gasto promedio por cliente.

## Integrantes

- **Alumno 1:** Juan Pablo Castelino
- **Alumno 2:** Walter Gabriel Sotelo
- **Alumno 3:** Esteban Leonardo Torres Martín

- **Airflow** orquesta la ingesta diaria del dataset y el reentrenamiento que esta dispara.
- **MinIO** almacena los datasets y los artefactos físicos del modelo mediante S3.
- **MLflow** registra parametros, métricas y versiones, y actúa como acceso al modelo.
- **FastAPI** expone la predicción por HTTP en dos instancias, `api-blue` y `api-green`.
- **Traefik** es el proxy que envía el tráfico a una de las dos y permite desplegar modelos nuevos sin cortar el servicio.
- **Docker Compose** ejecuta el entorno local completo.

FastAPI carga `purchase-spend-regressor@production` desde MLflow; MLflow recupera su artefacto desde MinIO. Cada vez que el pipeline promueve una versión, el DAG `deploy_purchase_model` la despliega con Blue-Green (ver [Despliegue Blue-Green](#despliegue-blue-green)).

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

`Airflow → MLflow Registry → MinIO (artefactos del modelo) → FastAPI (blue / green) ← Traefik :8000 ← clientes`

Airflow publica el dataset en MinIO, entrena y registra cada run en MLflow. MLflow conserva la metadata del experimento y publica los artefactos del modelo en el bucket `mlflow-artifacts` de MinIO. Las instancias de FastAPI se conectan únicamente a MLflow, mediante el alias estable `production` o la versión que les indique el deploy. Los clientes solo ven el proxy.

## Estructura

```text
demo-mlops/
├── airflow/       # Imagen propia, DAGs y dependencias del orquestador
├── api/           # API REST de inferencia con FastAPI
├── mlflow/        # Imagen del servidor de tracking
├── proxy/         # Configuración de Traefik y plantilla del color activo
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
- API REST (a través del proxy): http://localhost:8000/docs

Al iniciar el stack con `docker-compose up --build -d`, el servicio `airflow-trigger-ingest` espera a que Airflow cargue los tres DAGs, los habilita y dispara `ingest_purchase_dataset`. En la primera instalación crea el bucket `mlops-data`, publica el dataset y dispara el entrenamiento. En cada `docker-compose up` se vuelve a disparar la ingesta; el hash SHA-256 evita republicar el archivo o reentrenar si no cambió. Además, la ingesta conserva su schedule diario a medianoche, hora de Argentina.

En la primera instalación, `api-blue` y `api-green` esperan a que MLflow publique el alias `production` antes de iniciar. Cuando el entrenamiento lo publica, ambas APIs cargan el modelo y el DAG de deploy espera a que estén saludables antes de continuar.

`train_purchase_regressor` ejecuta estas etapas:

1. `eda_validate`: valida columnas, nulos, compras inválidas y consistencia demográfica; guarda el informe en `reports/eda_dataset_compras.json`.
2. `prepare_curated_dataset`: limpia transacciones, agrega por cliente y genera features sin leakage; guarda `curated/tabla_clientes_modelado.csv`.
3. `train_evaluate_and_register`: descarga exclusivamente la tabla curada desde MinIO, entrena la MLP y registra la ejecución en MLflow. Si `r2_test` es mayor o igual que el de `production`, promueve la versión con los alias `candidate` y `production`; de lo contrario, conserva la versión en producción.
4. `continue_if_promoted` y `announce_new_model`: si la versión se promovió, emiten el Dataset `models:/purchase-spend-regressor@production`, que dispara `deploy_purchase_model`. Si no se promovió, quedan en `skipped`.

La red replica la arquitectura final del TP: tres capas ocultas de 32 neuronas, activación ELU, dropout de 0.2, Adam, weight decay y early stopping. Al finalizar, MLflow muestra el experimento `purchase-spend-regression`, las métricas `r2_test`, `rmse_test`, `previous_production_r2_test` e `is_promoted`. El archivo físico del modelo queda en `s3://mlflow-artifacts`.

## API de predicción

Cada instancia carga al iniciar la versión a la que apunta el alias `production`; si el alias todavía no existe, no arranca. Después, las versiones nuevas le llegan mediante el deploy, sin reiniciarla. Envía las features por nombre a `POST /predict`; la API las ordena internamente según el esquema del modelo:

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

`GET /model-info` expone las columnas requeridas. La API responde `422` si faltan features o se envían nombres desconocidos. `GET /health` indica qué instancia respondió (`color`) y la versión de modelo cargada.

## Despliegue Blue-Green

La API corre en dos instancias, `api-blue` y `api-green`, y el proxy (Traefik, puerto 8000) envía todo el tráfico a una sola. Cuando el entrenamiento promueve una versión, `deploy_purchase_model` la pone en servicio sin cortar las predicciones:

```text
resolve_target → load_on_idle → smoke_test_idle → switch_traffic → verify_public
```

1. `resolve_target`: lee en MLflow la versión con alias `production` y, en `proxy-dynamic/active.yml`, qué color recibe el tráfico.
2. `load_on_idle`: carga esa versión en la instancia inactiva con `POST /admin/load`. La instancia valida el modelo antes de ponerlo en servicio; si falla, conserva el anterior.
3. `smoke_test_idle`: verifica `/health`, `/model-info` y una predicción en la instancia inactiva, que todavía no recibe tráfico.
4. `switch_traffic`: reescribe `active.yml` desde `proxy/active.yml.template`. Traefik lo recarga solo, sin reiniciarse ni cortar conexiones.
5. `verify_public`: confirma a través del proxy que se sirve la versión nueva. Si no ocurre en 30 segundos, devuelve el tráfico al color anterior y marca el DAG como fallido.

La instancia que deja de recibir tráfico conserva el modelo anterior. Casos que contempla:

| Situación | Resultado |
|---|---|
| Ninguna instancia corriendo (primera instalación) | `skipped`: al levantarlas cargarán `production` |
| Las dos corriendo | Despliegue Blue-Green |
| Solo una corriendo | Falla: no se puede desplegar sin cortar el servicio; hay que levantar la otra |
| La instancia activa ya sirve esa versión | `skipped`: no hay nada que desplegar |

**El alias `production` manda:** el deploy siempre despliega la versión a la que apunta, que es también la que cargan las instancias al reiniciarse. Para volver atrás, reasigna `production` a la versión anterior en MLflow (pestaña *Models*) y dispara `deploy_purchase_model` con **Trigger DAG**. Como la instancia inactiva todavía tiene cargada esa versión, el cambio es inmediato.

`POST /admin/load` exige el token `ADMIN_TOKEN` y el proxy no lo publica: solo Airflow lo alcanza por la red interna. En esta demo el token está en `docker-compose.yml`; en un entorno real iría en un gestor de secretos.

## Dependencias aisladas

Airflow solo instala `virtualenv`. Cada tarea corre en su propio entorno virtual: la ingesta usa `boto3` y `gdown`, EDA y preparación usan `boto3` y `pandas`, y el entrenamiento instala PyTorch, MLflow y scikit-learn. Esto evita mezclar las librerías de ML con las de Airflow.

Los entornos se guardan en el volumen `airflow-venv-cache` (`venv_cache_path` en los DAGs). Airflow crea cada uno la primera vez que lo necesita, identificado por un hash de sus requisitos, y lo reutiliza en las corridas siguientes, incluso si se recrean los contenedores. Por eso solo la primera ejecución demora más. Si cambian los requisitos de una tarea, Airflow crea un entorno nuevo automáticamente. Para forzar una reinstalación completa, con el stack detenido: `docker volume rm <proyecto>_airflow-venv-cache` (el prefijo es el nombre de la carpeta del repo, por defecto `machine_learning_operations`).

## Detener

```bash
docker-compose down
```

Los datos persisten en volumenes Docker. Para eliminarlos intencionalmente: `docker-compose down --volumes`.
