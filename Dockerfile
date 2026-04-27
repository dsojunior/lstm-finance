# Precisei explicitar a versão da plataforma pra rodar no MacOS
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY model/ ./model/

# Expõe a porta da API para 5001, para evitar conflito de portas no MacOS
EXPOSE 5001

CMD ["python", "api/app.py"]