# запуск проекта

## Запускаем neo4j
    sudo docker-compose up -d --build

## запуск llama 

В первый раз:  

    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama  
    docker exec -it ollama ollama run llama3:8b  

Для перезапуска:

    sudo docker start ollama

## создаём виртуальное окружение и устанавливаем необходимые зависимости
    
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements_default.txt

## Запускаем pipline
    python3 kg_builder.py

## Запускаем fastapi сервер 

    ВЫПОЛНИТЬ В ОТДЕЛЬНОМ ТЕРМИНАЛЕ, ПРЕДВАРИТЕЛЬНО ЗАПУСТИВ ТАМ ОКРУЖЕНИЕ ЧЕРЕЗ:
    source venv/bin/activate

    uvicorn server:app --host 0.0.0.0 --port 8000

## запускаем бота
    
    ВЫПОЛНИТЬ В ОТДЕЛЬНОМ ТЕРМИНАЛЕ, ПРЕДВАРИТЕЛЬНО ЗАПУСТИВ ТАМ ОКРУЖЕНИЕ ЧЕРЕЗ:
    source venv/bin/activate
    python3 bot.py
