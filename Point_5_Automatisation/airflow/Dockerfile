# airflow/Dockerfile
FROM apache/airflow:2.9.2-python3.11

# Passer en root pour installer des paquets système
USER root

# Installer gcc/g++ et dépendances système pour psycopg2, xgboost, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Revenir à l'utilisateur airflow
USER airflow

# Définir AIRFLOW_HOME
ENV AIRFLOW_HOME=/opt/airflow

COPY airflow/dags/ /opt/airflow/dags/
COPY airflow/requirements.txt /opt/airflow/requirements.txt

COPY streaming/ /opt/airflow/streaming/


COPY etl/ /opt/airflow/etl/


COPY ml/models/ /opt/airflow/ml_models/
COPY ./streaming/requirements.txt /opt/airflow/streaming_requirements.txt
COPY etl/requirements.txt /opt/airflow/etl_requirements.txt

# Installer les packages Python nécessaires
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r ${AIRFLOW_HOME}/requirements.txt \
    && pip install --no-cache-dir -r ${AIRFLOW_HOME}/streaming_requirements.txt \
    && pip install --no-cache-dir -r ${AIRFLOW_HOME}/etl_requirements.txt

# Créer les dossiers nécessaires
RUN mkdir -p ${AIRFLOW_HOME}/logs

# Définir le dossier de travail
WORKDIR ${AIRFLOW_HOME}

# Commande par défaut pour dev : webserver + scheduler
CMD ["bash", "-c", "\
    airflow db init && \
    airflow users create \
        --username ${AIRFLOW_USER} \
        --password ${AIRFLOW_PASSWORD} \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com ; \
    airflow scheduler & airflow webserver"]