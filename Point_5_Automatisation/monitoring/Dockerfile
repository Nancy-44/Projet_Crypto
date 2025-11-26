FROM python:3.11-slim

WORKDIR /app

COPY crypto_metrics.py .

RUN pip install requests prometheus_client

EXPOSE 9200

CMD ["python3", "crypto_metrics.py"]
