"""Simula la publicación diaria del dataset de compras por el equipo de BI.

Descarga el CSV desde la ruta remota, lo publica en MinIO solo si su SHA-256
cambió y, en ese caso, emite el Dataset que dispara el reentrenamiento.
"""

import pendulum
from airflow.decorators import dag, task

from purchase_datasets import RAW_DATASET, RAW_DATASET_KEY


ARGENTINA_TIMEZONE = pendulum.timezone("America/Argentina/Buenos_Aires")
# Cada venv se crea una sola vez por combinacion de requisitos y se reutiliza
# entre corridas. El directorio es un volumen de Docker (ver docker-compose.yml).
VENV_CACHE_PATH = "/opt/airflow/venv-cache"


@dag(
    dag_id="ingest_purchase_dataset",
    description="Publica el dataset de compras en MinIO cuando cambia su contenido.",
    schedule="@daily",
    start_date=pendulum.datetime(2025, 1, 1, tz=ARGENTINA_TIMEZONE),
    catchup=False,
    max_active_runs=1,
    tags=["demo", "mlops", "minio", "ingesta"],
)
def ingest_purchase_dataset():
    @task.virtualenv(
        requirements=["boto3==1.35.71", "gdown==5.2.0"],
        system_site_packages=False,
        venv_cache_path=VENV_CACHE_PATH,
    )
    def download_and_publish(dataset_key: str) -> bool:
        """Publica el CSV remoto si difiere del publicado; devuelve si hubo cambios.

        El hash del archivo publicado se guarda como metadata del propio objeto
        en MinIO, por lo que no hace falta un archivo de estado aparte.
        """
        import hashlib
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import boto3
        import gdown
        from botocore.exceptions import ClientError

        source_url = os.getenv("DATASET_SOURCE_URL")
        if not source_url:
            raise ValueError("Falta la variable de entorno DATASET_SOURCE_URL")

        bucket = os.getenv("S3_BUCKET", "mlops-data")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            client.create_bucket(Bucket=bucket)

        with TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset_compras.csv"
            if gdown.download(url=source_url, output=str(dataset_path), fuzzy=True) is None:
                raise RuntimeError(f"No se pudo descargar el dataset desde {source_url}")
            if dataset_path.stat().st_size == 0:
                raise ValueError("El dataset descargado esta vacio")
            dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

            try:
                published = client.head_object(Bucket=bucket, Key=dataset_key)
                published_sha256 = published["Metadata"].get("sha256")
            except ClientError as error:
                if error.response["Error"]["Code"] != "404":
                    raise
                published_sha256 = None

            if dataset_sha256 == published_sha256:
                print(f"Dataset sin cambios (SHA-256 {dataset_sha256}); no se publica.")
                return False

            client.upload_file(
                str(dataset_path),
                bucket,
                dataset_key,
                ExtraArgs={"Metadata": {"sha256": dataset_sha256, "source_url": source_url}},
            )
            print(
                f"Dataset publicado en s3://{bucket}/{dataset_key} "
                f"(SHA-256 {dataset_sha256}, anterior {published_sha256})."
            )
            return True

    @task.short_circuit
    def continue_if_changed(changed: bool) -> bool:
        """Omite el aviso de dataset nuevo cuando el contenido no cambió."""
        return changed

    @task(outlets=[RAW_DATASET])
    def announce_new_dataset() -> None:
        """Al terminar con éxito, Airflow registra un evento sobre RAW_DATASET."""

    changed = download_and_publish(RAW_DATASET_KEY)
    continue_if_changed(changed) >> announce_new_dataset()


ingest_purchase_dataset()
