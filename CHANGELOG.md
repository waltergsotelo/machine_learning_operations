# Changelog

Este archivo registra los cambios relevantes del proyecto por entrega.

## Proxima entrega

### Pendiente

- Separar la ingesta del dataset en un proceso ETL externo al DAG de entrenamiento.
- Consumir el dataset crudo exclusivamente desde S3 o MinIO.
- Agregar pruebas automatizadas, CI/CD y observabilidad del pipeline.
- Externalizar credenciales y configuraciones sensibles.

## Entrega actual - 2026-10-05

### Agregado

- Descarga automatica de `dataset_compras.csv` desde su origen publico en Google Drive.
- Deteccion de cambios mediante SHA-256 antes de ejecutar EDA y entrenamiento.
- Estado persistente en `state/dataset_compras.json` con el hash y la corrida procesada.
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
