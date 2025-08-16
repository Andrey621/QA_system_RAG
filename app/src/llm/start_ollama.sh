#!/bin/sh

wait_for_ollama() {
    echo "Ожидание запуска сервера Ollama..."
    while ! curl -s http://localhost:11434/api/version > /dev/null; do
        sleep 1
    done
    echo "Сервер Ollama запущен."
}

echo "Запуск сервера Ollama в фоновом режиме..."
ollama serve &

wait_for_ollama

if ! ollama list | grep -q "qwen:0.5b"; then
    echo "Загрузка модели qwen:0.5b..."
    ollama pull qwen:0.5b
else
    echo "Модель qwen:0.5b уже загружена."
fi

echo "Сервер Ollama и модель готовы к работе."
tail -f /dev/null