#!/bin/sh

MODEL_TAG="marketing-train-v1"
ADAPTER_PATH="/app/models/fine_tuned/marketing-train-v1"

wait_for_ollama() {
    echo "Ожидание запуска сервера Ollama..."
    while ! curl -s http://localhost:11434/api/version > /dev/null; do
        sleep 1
    done
    echo "Сервер Ollama запущен."
}

ensure_model_created() {
    if ollama list | grep -q "$MODEL_TAG"; then
        echo "Модель $MODEL_TAG уже существует."
        return
    fi
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "Папка адаптера $ADAPTER_PATH не найдена. Пропускаем создание модели."
        return
    fi
    cat <<EOF >/tmp/Modelfile
FROM qwen:0.5b
ADAPTER $ADAPTER_PATH
EOF
    echo "Создание модели $MODEL_TAG на основе $ADAPTER_PATH..."
    ollama create "$MODEL_TAG" -f /tmp/Modelfile
}

echo "Запуск сервера Ollama..."
ollama serve &
wait_for_ollama

if ! ollama list | grep -q "qwen:0.5b"; then
    echo "Загрузка базовой модели qwen:0.5b..."
    ollama pull qwen:0.5b
fi

ensure_model_created

echo "Сервер Ollama готов."
tail -f /dev/null
