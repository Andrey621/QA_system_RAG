# Используем базовый образ Python
FROM python:3.10-slim

# Установка зависимостей
WORKDIR /app
COPY requirements_fastapi.txt requirements_fastapi.txt
RUN pip install --no-cache-dir -r requirements_fastapi.txt

COPY req_fa2.txt req_fa2.txt
RUN pip install --no-cache-dir -r req_fa2.txt

# Копируем исходный код
COPY server.py .

# Открываем порт и запускаем сервер
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
