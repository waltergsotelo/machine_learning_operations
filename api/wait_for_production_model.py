"""Wait until MLflow publishes the configured model alias before starting the API."""

import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")
model_name = os.getenv("MODEL_NAME", "purchase-spend-regressor")
model_alias = os.getenv("MODEL_ALIAS", "production")
query = urlencode({"name": model_name, "alias": model_alias})
alias_url = f"{tracking_uri}/api/2.0/mlflow/registered-models/alias?{query}"

attempt = 0
while True:
    try:
        with urlopen(alias_url, timeout=5) as response:
            if response.status == 200:
                print(
                    f"MLflow publicó {model_name}@{model_alias}; iniciando la API.",
                    flush=True,
                )
                break
    except (HTTPError, URLError, TimeoutError) as error:
        attempt += 1
        if attempt % 12 == 1:
            print(
                f"Esperando {model_name}@{model_alias} en MLflow ({error}).",
                flush=True,
            )
    time.sleep(5)
