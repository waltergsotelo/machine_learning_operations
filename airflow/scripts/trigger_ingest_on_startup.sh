#!/usr/bin/env bash
set -euo pipefail

dag_ids=(
  ingest_purchase_dataset
  train_purchase_regressor
  deploy_purchase_model
)

loaded=false
for attempt in $(seq 1 60); do
  dag_list="$(airflow dags list -o yaml 2>/dev/null || true)"
  loaded=true
  for dag_id in "${dag_ids[@]}"; do
    if ! grep -Fq "dag_id: ${dag_id}" <<<"${dag_list}"; then
      loaded=false
      break
    fi
  done
  if [[ "${loaded}" == true ]]; then
    break
  fi
  sleep 2
done

if [[ "${loaded}" != true ]]; then
  echo "Airflow did not load all required DAGs within 120 seconds." >&2
  exit 1
fi

for dag_id in "${dag_ids[@]}"; do
  airflow dags unpause "${dag_id}"
done

airflow dags trigger ingest_purchase_dataset
