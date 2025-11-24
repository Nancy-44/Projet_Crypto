from airflow import DAG
from airflow.operators.python import PythonOperator 

from datetime import datetime, timedelta
import os, requests, subprocess
from dotenv import load_dotenv


load_dotenv("/opt/airflow/.env")  # chemin vers ton .env monté

API_URL = "http://api:8000/health"


# ---- Task : API Health Check ----
def check_api_health():
    try:
        r = requests.get(API_URL, timeout=5)
        if r.status_code != 200:
            raise Exception("API not healthy")
    except Exception as e:
        raise Exception(f"API unreachable: {e}")


# ---- Task : Run ETL ----
def run_etl():
    subprocess.run(["python3", "API_hist_to_postgres.py"], cwd="/opt/airflow/etl", check=True)


# ---- Task : Run Streaming ----
def run_streaming():
    subprocess.run(["python3", "stream.py"], cwd="/opt/airflow/streaming", check=True)




with DAG(
    dag_id="crypto_pipeline",
    description="Pipeline complet (ETL + Streaming + API health)",
    #schedule_interval="*/10 * * * *", # toutes les 10 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=15)
    }
) as dag:


    # Vérification API
    task_health = PythonOperator(
    task_id="check_api",
    python_callable=check_api_health
    )


    # ETL Historique
    task_etl = PythonOperator(
    task_id="run_etl_historical",
    python_callable=run_etl
    )


    # Streaming Temps-réel
    task_stream = PythonOperator(
    task_id="run_streaming",
    python_callable=run_streaming
    )


    task_health >> [task_etl, task_stream]