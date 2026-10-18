"""Datasets de Airflow compartidos entre la ingesta, el entrenamiento y el deploy.

Airflow conecta productor y consumidor por la URI exacta del Dataset, por eso
se definen una sola vez y los DAGs las importan.
"""

import os

from airflow.datasets import Dataset

RAW_DATASET_KEY = "raw/dataset_compras.csv"
RAW_DATASET = Dataset(f"s3://{os.getenv('S3_BUCKET', 'mlops-data')}/{RAW_DATASET_KEY}")

# Hay una versión nueva con el alias production. Usa la URI de modelos de MLflow.
MODEL_NAME = "purchase-spend-regressor"
PRODUCTION_MODEL = Dataset(f"models:/{MODEL_NAME}@production")
