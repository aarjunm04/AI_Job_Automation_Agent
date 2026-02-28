FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY rag_systems/ ./rag_systems/

EXPOSE 8090

CMD ["uvicorn", "rag_systems.production_server:app", "--host", "0.0.0.0", "--port", "8090"]

