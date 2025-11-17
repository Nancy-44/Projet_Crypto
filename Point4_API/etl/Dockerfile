FROM python:3.12-slim

WORKDIR /app

# Copier les dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code ETL
COPY . .

# Exécuter le script ETL
CMD ["python3", "API_hist_to_postgres.py"]
