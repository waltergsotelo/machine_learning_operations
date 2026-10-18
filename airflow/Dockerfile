FROM apache/airflow:2.10.5-python3.11

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Directorio de la cache de venvs de las tareas. Se crea con el usuario airflow
# para que el volumen montado encima herede ese dueno y sea escribible.
RUN mkdir -p /opt/airflow/venv-cache
