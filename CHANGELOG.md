# Changelog

Este archivo registra los cambios relevantes del proyecto por entrega.

## Proxima entrega

### Pendiente

- Separar la ingesta del dataset en un proceso ETL externo al DAG de entrenamiento.
- Consumir el dataset crudo exclusivamente desde S3 o MinIO.
- Agregar pruebas automatizadas, CI/CD y observabilidad del pipeline.
- Externalizar credenciales y configuraciones sensibles.

## Cambios sin publicar - 2026-10-15

Cambios locales respecto del ultimo commit `9169335` (`fix: address first deployment feedback`).

### Modificado

- Reemplazadas las imagenes `quay.io/minio/minio:latest` y `quay.io/minio/mc:latest` por `coollabsio/minio:RELEASE.2025-04-22T22-12-26Z` (ésto viene del hands-on de la 5ta clase) ante errores de acceso a los registros originales. La misma imagen de terceros aporta el servidor y el cliente `mc` para `minio-init`.
- Comentada la opcion de build `network: host` en MLflow, API y Airflow para utilizar la red predeterminada del builder sin requerir permisos adicionales. Esto solo afecta la construcción de las imágenes, no publica puertos ni cambia la red de los contenedores cuando ejecutás el pipeline.
- Restringida SQLAlchemy a `>=2.0,<2.1` en la imagen de MLflow y en los requisitos de la tarea de entrenamiento, manteniendo MLflow 2.17.2 y evitando el error de importacion de `FallbackAsyncAdaptedQueuePool`.
- Actualizado el README con las decisiones de compatibilidad, el comando de inicio para PowerShell, la reconstruccion de MLflow y los pasos de verificacion manual.

### Verificado

- Corrida local completada con registro y promocion de la version 1 de `purchase-spend-regressor`.
- Segunda corrida exitosa con entrenamiento omitido al detectar un dataset sin cambios.
- El servicio de inferencia y el escenario de cambio de dataset no forman parte de esta verificacion.

## Entrega - 2026-10-05

### Agregado

- Descarga automatica de `dataset_compras.csv` directamente desde la fuente publica en Google Drive configurada mediante `DATASET_SOURCE_URL`, usando `gdown` y una carpeta temporal en Airflow. El archivo se publica en `s3://mlops-data/raw/dataset_compras.csv` cuando se detectan cambios.
- Calculo del hash SHA-256 del archivo descargado y comparacion con el hash del ultimo dataset procesado correctamente, almacenado en MinIO en `state/dataset_compras.json`.
- Reentrenamiento solo cuando cambia el contenido del archivo o no existe un estado previo. Si el hash coincide, se omiten la validacion, la preparacion, el entrenamiento y la actualizacion del estado.
- Estado persistente en `state/dataset_compras.json` con el hash y la corrida procesada, actualizado solo despues de un entrenamiento exitoso para permitir reintentar el mismo dataset si el pipeline falla.
- Limite de una corrida activa del DAG para evitar reentrenamientos simultaneos.
- Documentacion del problema, dataset, arquitectura, ejecucion y alcance productivo.
- Secuencia de inicio reproducible que levanta la API despues de crear el primer modelo.

### Modificado

- Se reemplazo la dependencia de una ruta local externa por `DATASET_SOURCE_URL`.
- El reentrenamiento se omite cuando el contenido del dataset no cambia.
- Las instrucciones de ejecucion ahora parten desde la raiz real del repositorio.

## Entrega inicial - 2026-09-23

### Agregado

- Entorno local con Airflow, MinIO, MLflow, PostgreSQL y FastAPI mediante Docker Compose.
- Pipeline de validacion, preparacion, entrenamiento y registro de una MLP en PyTorch.
- Promocion de modelos mediante los alias `candidate` y `production` de MLflow.
- API de inferencia con features nominales, validacion de entrada y consulta de metadata.
- Documentacion de integrantes y comandos basicos del proyecto.
