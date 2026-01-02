FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    iputils-ping \
    curl \
    build-essential \
    pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY models.py .
COPY fine_tuning.py .
COPY training_jobs.py .
COPY server.py .


EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
