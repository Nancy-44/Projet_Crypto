FROM python:3.11-slim

WORKDIR /api

# Installer les dépendances systèmes pour psycopg2, xgboost, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgomp1 \
    libpq-dev \
&& rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip setuptools wheel

COPY ml/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy code api et models
COPY api/app ./app

COPY ml/models /models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
