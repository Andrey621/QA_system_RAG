# Используем базовый образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements_kg_builder.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements_kg_builder.txt

# Копируем зависимости
COPY req.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r req.txt


# Копируем ваш .py файл
COPY kg_builder.py .

# Копируем книгу "Котлер_Латеральный_маркетинг.doc"
COPY Котлер_Латеральный_маркетинг.pdf /app/Котлер_Латеральный_маркетинг.pdf

# Команда для запуска вашего скрипта
CMD ["python", "kg_builder.py"]