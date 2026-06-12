from datetime import datetime, timedelta
import json
import urllib.request

from airflow import DAG
from airflow.operators.python import PythonOperator


API_BASE_URL = "http://flat-price-api:8000"


def call_get_endpoint(path):
    url = f"{API_BASE_URL}{path}"

    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read().decode("utf-8")

    print(data)

    try:
        return json.loads(data)
    except Exception:
        return data


def call_post_endpoint(path, timeout=900):
    url = f"{API_BASE_URL}{path}"

    request = urllib.request.Request(
        url=url,
        data=b"",
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read().decode("utf-8")

    print(data)

    try:
        return json.loads(data)
    except Exception:
        return data


def check_api_before():
    return call_get_endpoint("/health")


def refresh_drift_report():
    return call_post_endpoint("/admin/run-drift-report", timeout=300)


def run_retraining():
    return call_post_endpoint("/admin/retrain", timeout=900)


def reload_champion_model():
    return call_post_endpoint("/admin/reload-model", timeout=300)


def check_api_after():
    return call_get_endpoint("/health")


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="flat_price_retraining_pipeline",
    description="Pipeline retrainingu modelu cen mieszkań",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 */6 * * *",
    catchup=False,
    tags=["mlops", "retraining", "flat-price"],
) as dag:

    task_check_api_before = PythonOperator(
        task_id="check_api_before",
        python_callable=check_api_before,
    )

    task_refresh_drift_report = PythonOperator(
        task_id="refresh_drift_report",
        python_callable=refresh_drift_report,
    )

    task_run_retraining = PythonOperator(
        task_id="run_retraining",
        python_callable=run_retraining,
    )

    task_reload_champion_model = PythonOperator(
        task_id="reload_champion_model",
        python_callable=reload_champion_model,
    )

    task_check_api_after = PythonOperator(
        task_id="check_api_after",
        python_callable=check_api_after,
    )

    (
        task_check_api_before
        >> task_refresh_drift_report
        >> task_run_retraining
        >> task_reload_champion_model
        >> task_check_api_after
    )
