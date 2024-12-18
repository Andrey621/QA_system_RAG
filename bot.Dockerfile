FROM python:3.10-slim

WORKDIR /app
COPY requirements_bot.txt requirements_bot.txt
RUN pip install --no-cache-dir -r requirements_bot.txt

COPY bot.py .

CMD ["python", "bot.py"]
