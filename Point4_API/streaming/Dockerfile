FROM python:3.11-bullseye

WORKDIR /streaming

# Installer pip à jour
RUN pip install --upgrade pip

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installer les packages Python nécessaires
RUN pip install --no-cache-dir websocket-client pymongo

# Copier le script stream.py
COPY stream.py .

# Définir la commande par défaut
CMD ["python", "stream.py"]
