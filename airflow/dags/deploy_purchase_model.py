"""Despliega con Blue-Green el modelo que tiene el alias production.

La API corre en dos instancias, api-blue y api-green, detrás de un proxy
(Traefik) que envía todo el tráfico a una sola. La versión nueva se carga y se
prueba en la instancia inactiva y recién entonces el proxy cambia el tráfico
hacia ella, sin cortar el servicio. La otra instancia conserva el modelo
anterior.

El alias production de MLflow manda: este DAG siempre despliega la versión a
la que apunta. Para volver atrás, se reasigna el alias a la versión anterior y
se dispara este DAG a mano.
"""

import math
import os
import time
from pathlib import Path

import pendulum
import requests
import yaml
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException, AirflowSkipException

from purchase_datasets import MODEL_NAME, PRODUCTION_MODEL


ARGENTINA_TIMEZONE = pendulum.timezone("America/Argentina/Buenos_Aires")
MODEL_ALIAS = "production"
COLORS = ("blue", "green")
PROXY_URL = os.getenv("PROXY_URL", "http://proxy:8000")
API_STARTUP_TIMEOUT_SECONDS = 600
API_HEALTH_POLL_SECONDS = 5
# Plantilla del repo (montada de solo lectura) y archivo que vigila Traefik.
PROXY_TEMPLATE_PATH = Path("/opt/airflow/proxy/active.yml.template")
PROXY_CONFIG_PATH = Path("/opt/airflow/proxy-dynamic/active.yml")


def _api_url(color: str) -> str:
    return f"http://api-{color}:8000"


def _instance_health(color: str) -> dict | None:
    """Devuelve el /health de una instancia, o None si no está corriendo."""
    try:
        response = requests.get(f"{_api_url(color)}/health", timeout=5)
    except (requests.ConnectionError, requests.Timeout):
        return None
    response.raise_for_status()
    return response.json()


def _wait_for_instances() -> dict[str, dict]:
    """Espera a que Blue y Green carguen el alias production desde MLflow."""
    deadline = time.monotonic() + API_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        health = {color: _instance_health(color) for color in COLORS}
        if all(health.values()):
            return health
        unavailable = [color for color in COLORS if health[color] is None]
        unavailable_apis = ", ".join(f"api-{color}" for color in unavailable)
        print(f"Esperando a que estén saludables: {unavailable_apis}.")
        time.sleep(API_HEALTH_POLL_SECONDS)

    raise AirflowFailException(
        "api-blue y api-green no estuvieron saludables dentro de "
        f"{API_STARTUP_TIMEOUT_SECONDS} segundos; el modelo production puede "
        "no estar publicado en MLflow."
    )


def _production_version() -> str:
    response = requests.get(
        f"{os.environ['MLFLOW_TRACKING_URI']}/api/2.0/mlflow/registered-models/alias",
        params={"name": MODEL_NAME, "alias": MODEL_ALIAS},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["model_version"]["version"]


def _wait_for_production_version() -> str:
    """Espera a que MLflow publique el alias que activó este deploy."""
    deadline = time.monotonic() + API_STARTUP_TIMEOUT_SECONDS
    next_log = 0.0
    while time.monotonic() < deadline:
        try:
            version = _production_version()
            print(f"MLflow publicó {MODEL_NAME}@{MODEL_ALIAS}, versión {version}.")
            return version
        except requests.RequestException as error:
            now = time.monotonic()
            if now >= next_log:
                print(f"Esperando {MODEL_NAME}@{MODEL_ALIAS} en MLflow ({error}).")
                next_log = now + 30
            time.sleep(API_HEALTH_POLL_SECONDS)

    raise AirflowFailException(
        f"MLflow no publicó {MODEL_NAME}@{MODEL_ALIAS} dentro de "
        f"{API_STARTUP_TIMEOUT_SECONDS} segundos."
    )


def _active_color() -> str:
    config = yaml.safe_load(PROXY_CONFIG_PATH.read_text(encoding="utf-8"))
    return config["http"]["routers"]["api"]["service"].removeprefix("api-")


def _set_active_color(color: str) -> None:
    """Reescribe la configuración del proxy de forma atómica; Traefik la recarga sola."""
    content = PROXY_TEMPLATE_PATH.read_text(encoding="utf-8").replace("{color}", color)
    temporary_path = PROXY_CONFIG_PATH.with_name(f".{PROXY_CONFIG_PATH.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, PROXY_CONFIG_PATH)


@dag(
    dag_id="deploy_purchase_model",
    description="Despliega con Blue-Green el modelo production en la API sin cortar el servicio.",
    schedule=[PRODUCTION_MODEL],
    start_date=pendulum.datetime(2025, 1, 1, tz=ARGENTINA_TIMEZONE),
    catchup=False,
    max_active_runs=1,
    tags=["demo", "mlops", "fastapi", "blue-green"],
)
def deploy_purchase_model():
    @task
    def resolve_target() -> dict[str, str]:
        """Decide qué versión desplegar y en qué color, o si no hay nada que hacer."""
        version = _wait_for_production_version()
        health = _wait_for_instances()

        active = _active_color()
        if health[active]["model_version"] == version:
            raise AirflowSkipException(f"api-{active} ya sirve la versión {version}.")
        idle = next(color for color in COLORS if color != active)
        print(
            f"Desplegar versión {version} en api-{idle}; api-{active} sirve la versión "
            f"{health[active]['model_version']}."
        )
        return {"version": version, "active": active, "idle": idle}

    @task
    def load_on_idle(target: dict[str, str]) -> None:
        """Carga la versión en la instancia inactiva, que todavía no recibe tráfico."""
        response = requests.post(
            f"{_api_url(target['idle'])}/admin/load",
            json={"version": target["version"]},
            headers={"X-Admin-Token": os.environ["ADMIN_TOKEN"]},
            timeout=300,
        )
        if not response.ok:
            raise RuntimeError(
                f"api-{target['idle']} no pudo cargar la versión {target['version']}: "
                f"{response.status_code} {response.text}"
            )
        print(response.json())

    @task
    def smoke_test_idle(target: dict[str, str]) -> None:
        """Prueba la instancia inactiva con los mismos endpoints que usan los clientes."""
        url = _api_url(target["idle"])
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        if health.json()["model_version"] != target["version"]:
            raise ValueError(f"api-{target['idle']} reporta {health.json()}")

        model_info = requests.get(f"{url}/model-info", timeout=10)
        model_info.raise_for_status()
        sample = {column: 0.0 for column in model_info.json()["feature_columns"]}
        response = requests.post(f"{url}/predict", json={"features": sample}, timeout=10)
        response.raise_for_status()
        prediction = response.json()["prediction"]
        if not math.isfinite(prediction):
            raise ValueError(f"Predicción no finita: {prediction}")
        print(f"api-{target['idle']} responde con la versión {target['version']}: {prediction}")

    @task
    def switch_traffic(target: dict[str, str]) -> None:
        """Hace que el proxy envíe el tráfico a la instancia que se acaba de probar."""
        _set_active_color(target["idle"])
        print(f"Tráfico: api-{target['active']} -> api-{target['idle']}")

    @task
    def verify_public(target: dict[str, str]) -> None:
        """Confirma a través del proxy el cambio; si no ocurre, devuelve el tráfico."""
        deadline = time.monotonic() + 30
        last_response: object = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{PROXY_URL}/health", timeout=5)
                last_response = response.json() if response.ok else response.text
                if (
                    response.ok
                    and last_response["color"] == target["idle"]
                    and last_response["model_version"] == target["version"]
                ):
                    print(f"El proxy sirve la versión {target['version']} desde api-{target['idle']}.")
                    return
            except requests.RequestException as error:
                last_response = str(error)
            time.sleep(1)

        _set_active_color(target["active"])
        raise AirflowFailException(
            f"El proxy no sirve la versión {target['version']} desde api-{target['idle']} "
            f"(última respuesta: {last_response}). Se devolvió el tráfico a api-{target['active']}."
        )

    target = resolve_target()
    (
        load_on_idle(target)
        >> smoke_test_idle(target)
        >> switch_traffic(target)
        >> verify_public(target)
    )


deploy_purchase_model()
