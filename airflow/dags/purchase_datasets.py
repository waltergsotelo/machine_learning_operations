"""Datasets de Airflow compartidos entre la ingesta y el entrenamiento.

Airflow conecta productor y consumidor por la URI exacta del Dataset, por eso
se define una sola vez y ambos DAGs la importan.
"""

import os

from airflow.datasets import Dataset

RAW_DATASET_KEY = "raw/dataset_compras.csv"
RAW_DATASET = Dataset(f"s3://{os.getenv('S3_BUCKET', 'mlops-data')}/{RAW_DATASET_KEY}")
