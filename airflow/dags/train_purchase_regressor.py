"""Orquesta carga, EDA, preparación y entrenamiento aislado de una MLP."""

import pendulum
from airflow.decorators import dag, task


ARGENTINA_TIMEZONE = pendulum.timezone("America/Argentina/Buenos_Aires")


@dag(
    dag_id="train_purchase_regressor",
    description="ETL recurrente, entrenamiento aislado y promocion de la MLP en MLflow.",
    schedule_interval="*/30 * * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz=ARGENTINA_TIMEZONE),
    catchup=False,
    max_active_runs=1,
    tags=["demo", "mlops", "minio", "mlflow", "pytorch"],
)
def train_purchase_regressor():
    @task.virtualenv(
        requirements=["boto3==1.35.71", "gdown==5.2.0"],
        system_site_packages=False,
    )
    def download_raw_dataset() -> dict[str, object]:
        import hashlib
        import json
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import boto3
        import gdown
        from botocore.exceptions import ClientError

        dataset_source_url = os.getenv(
            "DATASET_SOURCE_URL",
            "https://drive.google.com/file/d/1e_FRepCpHBWWGj3TYG4HY7WHALCyR4wQ/view",
        )
        bucket = os.getenv("S3_BUCKET", "mlops-data")
        dataset_key = "raw/dataset_compras.csv"
        state_key = "state/dataset_compras.json"
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
            downloaded_path = gdown.download(
                url=dataset_source_url,
                output=str(dataset_path),
                quiet=False,
                fuzzy=True,
            )
            if downloaded_path is None or not dataset_path.is_file():
                raise RuntimeError(f"No se pudo descargar el dataset desde {dataset_source_url}")
            if dataset_path.stat().st_size == 0:
                raise ValueError("El dataset descargado esta vacio")

            digest = hashlib.sha256()
            with dataset_path.open("rb") as dataset_file:
                for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            dataset_sha256 = digest.hexdigest()

            previous_sha256 = None
            try:
                previous_state = json.loads(
                    client.get_object(Bucket=bucket, Key=state_key)["Body"].read()
                )
                previous_sha256 = previous_state.get("sha256")
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
                    raise

            changed = dataset_sha256 != previous_sha256
            if changed:
                client.upload_file(str(dataset_path), bucket, dataset_key)

            return {
                "changed": changed,
                "dataset_key": dataset_key,
                "sha256": dataset_sha256,
                "size_bytes": dataset_path.stat().st_size,
                "source_url": dataset_source_url,
                "state_key": state_key,
            }

    @task
    def continue_if_dataset_changed(dataset_info: dict[str, object]) -> str:
        from airflow.exceptions import AirflowSkipException

        if not dataset_info["changed"]:
            raise AirflowSkipException(
                f"Dataset sin cambios: SHA-256 {dataset_info['sha256']}"
            )
        return str(dataset_info["dataset_key"])

    @task.virtualenv(
        requirements=["boto3==1.35.71", "pandas==2.2.3"],
        system_site_packages=False,
    )
    def eda_validate(raw_dataset_key: str) -> str:
        import json
        import os
        from io import BytesIO

        import boto3
        import pandas as pd

        bucket = os.getenv("S3_BUCKET", "mlops-data")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        dataset = pd.read_csv(
            BytesIO(client.get_object(Bucket=bucket, Key=raw_dataset_key)["Body"].read())
        )
        required_columns = {
            "User_ID",
            "Product_ID",
            "Age",
            "Gender",
            "Marital_Status",
            "City_Category",
            "Stay_In_Current_City_Years",
            "Product_Category",
            "Purchase",
        }
        missing_columns = sorted(required_columns - set(dataset.columns))
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        if dataset.empty:
            raise ValueError("El dataset crudo no contiene filas")

        demographics = [
            "Age",
            "Gender",
            "Marital_Status",
            "City_Category",
            "Stay_In_Current_City_Years",
        ]
        report = {
            "source": f"s3://{bucket}/{raw_dataset_key}",
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns)),
            "unique_users": int(dataset["User_ID"].nunique()),
            "nulls_by_column": {
                column: int(count) for column, count in dataset.isna().sum().items() if count
            },
            "invalid_purchase_rows": int(
                pd.to_numeric(dataset["Purchase"], errors="coerce").isna().sum()
            ),
            "inconsistent_demographics": {
                column: int(count)
                for column, count in dataset.groupby("User_ID")[demographics]
                .nunique()
                .gt(1)
                .sum()
                .items()
                if count
            },
        }
        report_key = "reports/eda_dataset_compras.json"
        client.put_object(
            Bucket=bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return raw_dataset_key

    @task.virtualenv(
        requirements=["boto3==1.35.71", "pandas==2.2.3"],
        system_site_packages=False,
    )
    def prepare_curated_dataset(raw_dataset_key: str) -> str:
        import os
        from io import BytesIO

        import boto3
        import pandas as pd

        bucket = os.getenv("S3_BUCKET", "mlops-data")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        raw_dataset = pd.read_csv(
            BytesIO(client.get_object(Bucket=bucket, Key=raw_dataset_key)["Body"].read())
        )
        required_columns = [
            "User_ID",
            "Product_ID",
            "Age",
            "Gender",
            "Marital_Status",
            "City_Category",
            "Stay_In_Current_City_Years",
            "Product_Category",
            "Purchase",
        ]
        transactions = raw_dataset[required_columns].copy()
        transactions["Purchase"] = pd.to_numeric(transactions["Purchase"], errors="coerce")
        transactions = transactions.dropna(subset=required_columns)
        if transactions.empty:
            raise ValueError("No quedaron transacciones validas luego de la limpieza")

        demographics = [
            "Age",
            "Gender",
            "Marital_Status",
            "City_Category",
            "Stay_In_Current_City_Years",
        ]
        grouped = transactions.groupby("User_ID")
        customer_table = grouped[demographics].first()
        categories = sorted(transactions["Product_Category"].unique())
        category_mix = (
            pd.crosstab(
                transactions["User_ID"],
                transactions["Product_Category"],
                normalize="index",
            )
            .reindex(columns=categories, fill_value=0.0)
            .add_prefix("prop_cat_")
        )
        transaction_count = grouped.size()
        behavior = pd.DataFrame(
            {
                "n_transacciones": transaction_count,
                "n_categorias_distintas": grouped["Product_Category"].nunique(),
                "n_productos_distintos": grouped["Product_ID"].nunique(),
                "concentracion_hhi": (category_mix.to_numpy() ** 2).sum(axis=1),
            },
            index=category_mix.index,
        )
        behavior["tasa_recompra"] = (
            1 - behavior["n_productos_distintos"] / behavior["n_transacciones"]
        )
        behavior["categorias_por_compra"] = (
            behavior["n_categorias_distintas"] / behavior["n_transacciones"]
        )

        features = customer_table.join(category_mix).join(behavior)
        features["Gender"] = (features["Gender"] == "M").astype(int)
        features["City_nivel"] = features["City_Category"].map({"C": 1, "B": 2, "A": 3})
        features = features.drop(columns=["City_Category"])
        features = pd.get_dummies(
            features,
            columns=["Age", "Stay_In_Current_City_Years"],
            prefix=["edad", "antig"],
            drop_first=True,
            dtype="float32",
        ).astype("float32")
        target = grouped["Purchase"].mean().rename("Gasto_Promedio")
        curated_dataset = features.join(target).reset_index()
        curated_key = "curated/tabla_clientes_modelado.csv"
        client.put_object(
            Bucket=bucket,
            Key=curated_key,
            Body=curated_dataset.to_csv(index=False).encode("utf-8"),
            ContentType="text/csv",
        )
        return curated_key

    @task.virtualenv(
        requirements=[
            "boto3==1.35.71",
            "mlflow==2.17.2",
            "sqlalchemy>=2.0,<2.1",
            "numpy==1.26.4",
            "pandas==2.2.3",
            "scikit-learn==1.5.2",
            "torch==2.5.1",
        ],
        system_site_packages=False,
    )
    def train_evaluate_and_register(dataset_key: str) -> dict[str, object]:
        import os
        from copy import deepcopy
        from io import BytesIO

        import boto3
        import mlflow
        import mlflow.pytorch
        import numpy as np
        import pandas as pd
        import torch
        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        class MLPGasto(nn.Module):
            def __init__(self, n_features: int) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(n_features, 32),
                    nn.ELU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 32),
                    nn.ELU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 32),
                    nn.ELU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                )

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return self.network(features)

        class ServingRegressor(nn.Module):
            def __init__(
                self,
                network: MLPGasto,
                feature_mean: np.ndarray,
                feature_scale: np.ndarray,
                target_mean: float,
                target_scale: float,
            ) -> None:
                super().__init__()
                self.network = network
                self.register_buffer(
                    "feature_mean", torch.tensor(feature_mean, dtype=torch.float32)
                )
                self.register_buffer(
                    "feature_scale", torch.tensor(feature_scale, dtype=torch.float32)
                )
                self.register_buffer("target_mean", torch.tensor(target_mean, dtype=torch.float32))
                self.register_buffer(
                    "target_scale", torch.tensor(target_scale, dtype=torch.float32)
                )

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                normalized_features = (features - self.feature_mean) / self.feature_scale
                return self.network(normalized_features) * self.target_scale + self.target_mean

        seed = 42
        bucket = os.getenv("S3_BUCKET", "mlops-data")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("purchase-spend-regression")

        with mlflow.start_run(run_name="airflow-purchase-mlp") as run:
            response = client.get_object(Bucket=bucket, Key=dataset_key)
            dataset = pd.read_csv(BytesIO(response["Body"].read()))
            target_column = "Gasto_Promedio"
            feature_columns = [
                column for column in dataset.columns if column not in {"User_ID", target_column}
            ]
            features = dataset[feature_columns].to_numpy(dtype=np.float32)
            target = dataset[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

            features_train, features_remaining, target_train, target_remaining = train_test_split(
                features, target, test_size=0.4, random_state=seed
            )
            features_validation, features_test, target_validation, target_test = train_test_split(
                features_remaining, target_remaining, test_size=0.5, random_state=seed
            )
            feature_mean = features_train.mean(axis=0)
            feature_scale = features_train.std(axis=0)
            feature_scale[feature_scale == 0] = 1.0
            target_mean = float(target_train.mean())
            target_scale = float(target_train.std()) or 1.0

            train_dataset = TensorDataset(
                torch.tensor((features_train - feature_mean) / feature_scale),
                torch.tensor((target_train - target_mean) / target_scale),
            )
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            validation_features = torch.tensor(
                (features_validation - feature_mean) / feature_scale
            )
            validation_target = torch.tensor(
                (target_validation - target_mean) / target_scale
            )

            network = MLPGasto(len(feature_columns))
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_function = nn.MSELoss()
            best_validation_loss = float("inf")
            best_weights = None
            patience = 15
            epochs_without_improvement = 0
            epochs_trained = 0

            for epoch in range(1, 151):
                network.train()
                for batch_features, batch_target in train_loader:
                    optimizer.zero_grad()
                    loss = loss_function(network(batch_features), batch_target)
                    loss.backward()
                    optimizer.step()

                network.eval()
                with torch.no_grad():
                    validation_loss = loss_function(
                        network(validation_features), validation_target
                    ).item()
                epochs_trained = epoch
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_weights = deepcopy(network.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

            network.load_state_dict(best_weights)
            network.eval()
            serving_model = ServingRegressor(
                network, feature_mean, feature_scale, target_mean, target_scale
            ).eval()
            with torch.no_grad():
                test_predictions = serving_model(torch.tensor(features_test)).numpy().ravel()
            test_target = target_test.ravel()
            r2 = r2_score(test_target, test_predictions)
            rmse = float(mean_squared_error(test_target, test_predictions, squared=False))

            mlflow.log_params(
                {
                    "architecture": "32-32-32",
                    "activation": "ELU",
                    "dropout": 0.2,
                    "optimizer": "Adam",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 128,
                    "max_epochs": 150,
                    "early_stopping_patience": patience,
                    "feature_count": len(feature_columns),
                }
            )
            mlflow.log_metrics(
                {
                    "r2_test": r2,
                    "rmse_test": rmse,
                    "validation_loss": best_validation_loss,
                    "epochs_trained": epochs_trained,
                }
            )
            mlflow.set_tags(
                {
                    "orchestrator": "airflow",
                    "framework": "pytorch",
                    "dataset": f"s3://{bucket}/{dataset_key}",
                }
            )
            mlflow.log_dict({"feature_columns": feature_columns}, "feature_schema.json")
            traced_model = torch.jit.trace(
                serving_model, torch.tensor(features_train[:1], dtype=torch.float32)
            )
            mlflow.pytorch.log_model(traced_model, artifact_path="model")

            model_name = "purchase-spend-regressor"
            registry_client = MlflowClient()
            previous_r2 = None
            try:
                production_model = registry_client.get_model_version_by_alias(
                    model_name, "production"
                )
                previous_run = registry_client.get_run(production_model.run_id)
                previous_r2 = previous_run.data.metrics.get("r2_test")
            except MlflowException:
                pass

            is_promoted = previous_r2 is None or r2 >= previous_r2
            mlflow.set_tags(
                {
                    "promotion_status": "promoted" if is_promoted else "rejected",
                    "production_comparison": "first_model"
                    if previous_r2 is None
                    else "r2_test",
                }
            )
            if previous_r2 is not None:
                mlflow.log_metric("previous_production_r2_test", previous_r2)
            mlflow.log_metric("is_promoted", int(is_promoted))

            model_version = None
            if is_promoted:
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(model_uri, model_name)
                registry_client.set_registered_model_alias(
                    model_name, "candidate", registered_model.version
                )
                registry_client.set_registered_model_alias(
                    model_name, "production", registered_model.version
                )
                model_version = registered_model.version

            return {
                "run_id": run.info.run_id,
                "r2_test": r2,
                "rmse_test": rmse,
                "previous_production_r2_test": previous_r2,
                "is_promoted": is_promoted,
                "model_name": model_name,
                "model_version": model_version,
            }

    @task.virtualenv(
        requirements=["boto3==1.35.71"],
        system_site_packages=False,
    )
    def mark_dataset_processed(
        dataset_info: dict[str, object], training_result: dict[str, object]
    ) -> str:
        import json
        import os
        from datetime import datetime, timezone

        import boto3

        bucket = os.getenv("S3_BUCKET", "mlops-data")
        state_key = str(dataset_info["state_key"])
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        state = {
            "sha256": dataset_info["sha256"],
            "size_bytes": dataset_info["size_bytes"],
            "source_url": dataset_info["source_url"],
            "dataset_key": dataset_info["dataset_key"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": training_result["run_id"],
            "model_version": training_result["model_version"],
            "is_promoted": training_result["is_promoted"],
        }
        client.put_object(
            Bucket=bucket,
            Key=state_key,
            Body=json.dumps(state, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return state_key

    downloaded_dataset = download_raw_dataset()
    raw_dataset = continue_if_dataset_changed(downloaded_dataset)
    validated_dataset = eda_validate(raw_dataset)
    curated_dataset = prepare_curated_dataset(validated_dataset)
    training_result = train_evaluate_and_register(curated_dataset)
    mark_dataset_processed(downloaded_dataset, training_result)


train_purchase_regressor()
