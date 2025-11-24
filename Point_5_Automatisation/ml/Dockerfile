FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Mettre pip à jour
RUN pip install --upgrade pip setuptools wheel

# Copier requirements et installer les packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le dossier models uniquement (code ML)
COPY models/ ./models/

# Dossier streaming et API ne sont pas copiés ici, ils seront montés via docker-compose

# Commande par défaut
CMD ["tail", "-f", "/dev/null"]
