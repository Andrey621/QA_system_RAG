# GraphRAG Telegram Bot

Проект реализует Telegram-бота для взаимодействия с системой Retrieval-Augmented Generation (RAG), построенной на графовой базе данных Neo4j

Бот позволяет задавать вопросы по документам, сравнивать ответы локальной модели с Yandex GPT и просматривать информацию из справочника,
хранящегося в графе знаний

## Быстрый старт

Чтобы запустить проект и начать задавать вопросы:

1.  **Запустите сервисы**:
    Убедитесь, что Docker и Docker Compose установлены. Перейдите в корневую директорию проекта (где находится `docker-compose.yml`) и выполните:
    ```bash
    docker-compose up --build -d
    ```
    Эта команда соберет необходимые образы и запустит все контейнеры в фоновом режиме. 
    Дождитесь, пока все сервисы будут запущены и здоровы (healthy),
    особенно `neo4j_container` и `ollama_container`.

    Это может занять несколько минут при первом запуске, так как будут загружаться модели.


2.  **Добавьте свой PDF-документ**:
    Чтобы система могла отвечать на вопросы, ей нужен документ. Используйте `curl`, чтобы отправить PDF в API:
    ```bash
    curl -X POST "http://127.0.0.1:8000/process_pdf/" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "pdf_file=@/path/to/your/document.pdf;type=application/pdf"
    ```
    *   Замените `/path/to/your/document.pdf` на путь к вашему PDF-файлу на вашей локальной машине.
    *   Дождитесь завершения запроса. В логах `fastapi_container` (`docker-compose logs fastapi`) не должно быть ошибок
    
    Это может занять некоторое время, так как документ разбивается на части, создаются эмбеддинги и сохраняется в Neo4j.


3.  **(Опционально) Добавьте темы для справочника**:
    Если вы хотите использовать функцию справочника в боте:
    *   Откройте веб-интерфейс Neo4j в браузере: `http://localhost:7474`.
    *   Войдите, используя логин и пароль из вашего файла `.env.neo4j` (по умолчанию `neo4j` / `password`).
    *   В поле запроса Cypher выполните команду для создания узла темы:
        ```cypher
        CREATE (t:Topic {name: "Тема 1", description: "Описание темы 1", links: ["http://example.com/link1", "http://example.com/link2"]})
        RETURN t
        ```
        Повторите для других тем.

4.  **Начните использовать Telegram-бота**:
    *   Найдите своего бота в Telegram (по имени, указанному при создании через @BotFather)
    *   Отправьте команду `/start`
    *   Следуйте инструкциям в меню бота:
        *   Выберите "1" для вопросов к ИИ (по вашему документу)
        *   Выберите "2" для просмотра справочника (по добавленным темам)
        *   Выберите "3" для сравнения ответов локальной модели и Yandex GPT (если настроено)


## Архитектура

Проект состоит из нескольких сервисов, управляемых Docker Compose:

1.  **Neo4j**: Графовая база данных для хранения документов, извлеченных сущностей, связей и эмбеддингов
2.  **Ollama**: Сервис для запуска и обслуживания локальных LLM (например, Qwen)
3.  **FastAPI (Server)**: Бэкенд-сервис, реализующй API для:
    *   Обработки PDF-документов (извлечение текста, разбиение на части, создание эмбеддингов, сохранение в Neo4j)
    *   Ответов на вопросы с использованием RAG (поиск по Neo4j, генерация ответа через локальную LLM или Yandex GPT)
    *   Сравнения качества ответов локальной модели и Yandex GPT
    *   Предоставления данных для справочника
4.  **Telegram Bot (Bot)**: Интерфейс взаимодействия с пользователем через Telegram

## Возможности

*   **Обработка документов**: Загрузка PDF-файлов через API FastAPI (`/process_pdf`), извлечение информации и построение графа знаний в Neo4j
*   **Вопросы к ИИ**: Пользователи могут задавать вопросы в Telegram-боте. Бот пересылает запрос в FastAPI, который использует RAG для поиска
    релевантной информации в Neo4j и генерирует ответ с помощью локальной LLM (Ollama)
*   **Сравнение моделей**: Пользователи могут отправить вопрос для сравнения. FastAPI получает ответы как от локальной модели (Ollama),
    так и от Yandex GPT, вычисляет схожесть ответов (косинусное расстояние между эмбеддингами) и предоставляет пользователю оба ответа и метрику качества.
*   **Справочник**: Просмотр структурированной информации, хранящейся в Neo4j, например, списка тем и деталей по каждой теме

## Технологии

*   **Backend (FastAPI)**: Python, FastAPI, LangChain, Neo4j Driver, Sentence Transformers, Ollama Python Library, Yandex Cloud ML SDK
*   **Frontend (Bot)**: Python, Aiogram
*   **База данных**: Neo4j
*   **LLM**: Ollama (для локальной модели, например, Qwen)
*   **Deployment**: Docker, Docker Compose

## Начало работы

### Предварительные требования

*   Docker и Docker Compose установлены на вашей системе
*   Аккаунт и API-ключ в Yandex Cloud (для использования Yandex GPT)
*   Токен Telegram-бота (получается через [@BotFather](https://t.me/BotFather))

### Установка и запуск

1. **Настройка переменных окружения**:
    Создайте файлы `.env` для каждого сервиса в соответствующих каталогах или рядом с `docker-compose.yml`:
    *   `./app/src/server/.env`: Для FastAPI
        ```env
        NEO4J_URI=bolt://neo4j:7687
        NEO4J_USERNAME=neo4j
        NEO4J_PASSWORD=your_neo4j_password # Должен совпадать с NEO4J_AUTH
        OLLAMA_BASE_URL=http://ollama:11434
        YANDEX_API_KEY=your_yandex_cloud_api_key
        YANDEX_FOLDER_ID=your_yandex_cloud_folder_id
        YANDEX_ASSISTANT_ID=your_created_yandex_assistant_id # Создается скриптом create_assistant.py
        YANDEX_INDEX_FILE=path_or_id_to_your_indexed_file # Для Yandex Search Index Tool
        ```
    *   `./app/src/bot/.env`: Для Telegram бота
        ```env
        API_TOKEN=your_telegram_bot_token
        FASTAPI_URL=http://fastapi:8000 # URL FastAPI сервиса внутри Docker сети
        ```
    *   `.env.neo4j` (рядом с `docker-compose.yml`): Для Neo4j
        ```env
        NEO4J_AUTH=neo4j/your_neo4j_password # Формат: username/password
        NEO4J_APOC_EXPORT_FILE_ENABLED=true
        NEO4J_APOC_IMPORT_FILE_ENABLED=true
        NEO4J_APOC_IMPORT_FILE_USE_NEO4J_CONFIG=true
        NEO4J_DBMS_SECURITY_PROCEDURES_UNRESTRICTED=apoc.*
        ```

2. **Создание ассистента Yandex GPT** (если используется):
    *   Убедитесь, что `YANDEX_API_KEY` и `YANDEX_FOLDER_ID` заданы в `.env.fastapi`
    *   Запустите скрипт `create_assistant.py` (внутри контейнера `fastapi` или локально с установленными зависимостями):
        ```bash
        # Из каталога ./app/src/server
        python create_assistant.py
        ```
    *   Скопируйте выведенный `assistant_id` и добавьте его в `.env.fastapi` как `YANDEX_ASSISTANT_ID=...`.

3. **Запуск сервисов**:
    Выполните команду в корневом каталоге проекта (где находится `docker-compose.yml`):
    ```bash
    docker-compose up -d
    ```
    Эта команда соберет Docker-образы (если необходимо) и запустит все сервисы в фоновом режиме

4. **Инициализация моделей**:
    *   При первом запуске Ollama автоматически загрузит указанную модель (например, `qwen2:0.5b`)
    *   Модель `cointegrated/rubert-tiny` для создания эмбеддингов будет автоматически загружена FastAPI при первом обращении или при запуске

5. **Обработка документа**:
    Перед тем как задавать вопросы, необходимо загрузить PDF-документ в систему:
    ```bash
    curl -X POST "http://localhost:8000/process_pdf/" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "pdf_file=@path/to/your/document.pdf;type=application/pdf"
    ```
    Дождитесь завершения обработки (в логах `fastapi_container` не должно быть ошибок)

6. **Использование Telegram-бота**:
    Найдите вашего бота в Telegram по имени и нажмите "Start". Следуйте инструкциям в меню бота

### Остановка сервисов

Для остановки и удаления контейнеров выполните:
```bash
docker-compose down
```
Чтобы остановить и удалить контейнеры, сети и **тома** (включая данные Neo4j и модели Ollama):
```bash
docker-compose down -v
```


## API Endpoints (FastAPI)

После запуска документация API доступна по адресу: `http://localhost:8000/docs`

*   `POST /process_pdf/`: Загрузить и обработать PDF-документ
*   `POST /answer_question/`: Получить ответ на вопрос, используя данные из Neo4j и локальную LLM
*   `POST /compare_answers/`: Сравнить ответы локальной модели и Yandex GPT
*   `GET /get_topics/`: Получить список тем из справочника (Neo4j)
*   `GET /get_topic/{topic_name}`: Получить детали конкретной темы из справочника

## Команды бота

*   `/start`: Запустить бота и показать главное меню
*   `/menu`: Вернуться в главное меню в любой момент
*   В меню:
    *   `1`: Режим "Обратиться к нейросети" - задать вопрос
    *   `2`: Режим "Обратиться к справочнику" - просмотр тем
    *   `3`: Режим "Сравнить качество ответов" - задать вопрос для сравнения моделей


## Важные замечания

*   **Ресурсы**: Сервисы, особенно Ollama и Neo4j, могут потреблять значительные ресурсы (CPU, RAM). Убедитесь, что ваша система соответствует требованиям
*   **GPU (для Ollama)**: Если вы хотите использовать GPU для ускорения работы моделей в Ollama, убедитесь, что ваша система настроена для использования GPU в Docker (nvidia-docker/nvidia-container-toolkit)
    В `docker-compose.yml` уже добавлена конфигурация `deploy` для проброса GPU
