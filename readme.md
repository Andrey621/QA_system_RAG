# запуск проекта

    docker-compose up --build . 

# Вызов pipline для составления графа 

    curl -X POST "http://127.0.0.1:8000/kg_builder/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "pdf_file=@/home/user/Desktop/PyCharmProjects/graphRAG/test.pdf;type=application/pdf"

# Добавление темы для справочника: 

    1. Заходим в neo4j по ссылке http://localhost:7474 и логинимся по логину и паролю из app/server/models.env
    2. Добавляем инфу по книгеCREATE (t1:Topic {name: "Тема 1", description: "Описание темы 1", links: ["http://example.com"]})