FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY app.py /app/app.py
COPY wait_for_production_model.py /app/wait_for_production_model.py

CMD ["sh", "-c", "python /app/wait_for_production_model.py && exec uvicorn app:app --host 0.0.0.0 --port 8000"]
