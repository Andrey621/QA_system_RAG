# pylint: disable=line-too-long, missing-module-docstring, import-error, redefined-outer-name, broad-exception-caught, invalid-name, missing-function-docstring, unused-argument, unused-import, undefined-variable, raise-missing-from, no-else-return, no-else-raise
"""
GraphRAG API

API для обработки PDF, построения графов знаний и ответов на вопросы.
"""

import logging
import os
import re

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, status

from huggingface_hub import snapshot_download
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain_text_splitters import TokenTextSplitter
from neo4j import GraphDatabase

from sklearn.metrics.pairwise import cosine_similarity
from yandex_cloud_ml_sdk import YCloudML


from models import HealthCheckResponse, CompareAnswersResponse, QuestionRequest, AnswerQuestionResponse, \
    GetTopicsResponse, GetTopicResponse, ProcessPdfResponse

load_dotenv()

# --- Основные настройки подключения ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_ASSISTANT_ID = os.getenv("YANDEX_ASSISTANT_ID")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "cointegrated/rubert-tiny")
EMBEDDINGS_CACHE_DIR = os.getenv("EMBEDDINGS_CACHE_DIR", "/app/models")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen:0.5b")

# Валидация критических переменных
if not NEO4J_PASSWORD:
    raise ValueError("Переменная окружения NEO4J_PASSWORD не установлена.")
if not EMBEDDING_MODEL_NAME:
    raise ValueError("Переменная окружения EMBEDDING_MODEL_NAME не может быть пустой.")
if not OLLAMA_MODEL_NAME:
    raise ValueError("Переменная окружения OLLAMA_MODEL_NAME не может быть пустой.")

# Логгирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройка модели и эмбеддингов
MODELS_DIR = EMBEDDINGS_CACHE_DIR
MODEL_NAME = EMBEDDING_MODEL_NAME
# Очищаем имя модели от недопустимых символов для путей файловой системы (базовая очистка)
CLEAN_MODEL_NAME = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', EMBEDDING_MODEL_NAME)
MODEL_PATH = os.path.join(MODELS_DIR, CLEAN_MODEL_NAME)
os.makedirs(MODEL_PATH, exist_ok=True)
logger.info("Обеспечение доступности модели %s в %s...", MODEL_NAME, MODEL_PATH)
try:
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Загрузка модели {MODEL_NAME}...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_PATH)
        logger.info("Модель успешно загружена.")
    else:
        logger.info("Модель уже существует. Пропускаем загрузку.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    raise

logger.info("Инициализация HuggingFaceEmbeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    logger.info("Модель успешно загружена в HuggingFaceEmbeddings.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    raise

logger.info("Подключение к Neo4j...")
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    logger.info("Драйвер Neo4j успешно подключен.")
except Exception as exc:
    logger.error("Не удалось подключиться к Neo4j: %s", exc)
    raise

logger.info("Инициализация Langchain Neo4jGraph...")
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    logger.info("Langchain Neo4jGraph успешно инициализирован.")
except Exception as exc:
    logger.error("Не удалось инициализировать Langchain Neo4jGraph: %s", exc)
    raise

logger.info("Инициализация Ollama LLM...")
try:
    llm = OllamaLLM(
        model=OLLAMA_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        system="Ты работаешь с текстом на русском языке. Ты должен извлекать сущности и связи между ними, а также отвечать на вопросы на русском языке."
    )
    logger.info("Ollama LLM успешно инициализирована.")
except Exception as exc:
    logger.error("Не удалось инициализировать Ollama LLM: %s", exc)
    raise


logger.info("Инициализация Yandex cloud...")
try:
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY
    )
    logger.info("Ollama LLM успешно инициализирована.")
except Exception as exc:
    logger.error("Не удалось инициализировать Ollama LLM: %s", exc)
    raise


app = FastAPI(
    title="GraphRAG API",
    description="API для обработки PDF, построения графов знаний и ответов на вопросы.",
    version="1.0.0",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Дополнительные функции
def get_yandex_gpt_response(prompt: str) -> str:
    """
    Получить ответ от Яндекс GPT

    Args:
        prompt: Текст запроса

    Returns:
        Ответ от модели Яндекс GPT или сообщение об ошибке
    """
    try:
        logger.info("Отправка запроса к Yandex GPT...")
        assistant = sdk.assistants.get(YANDEX_ASSISTANT_ID)
        thread = sdk.threads.create()
        thread.write(prompt)
        run = assistant.run(thread)
        result = run.wait(poll_interval=0.05)
        answer_text = result._message.parts[0]
        logger.info("Получен ответ от Yandex GPT.")
        return answer_text
    except requests.exceptions.RequestException as exc:
        logger.error("Сетевая ошибка при запросе к Yandex GPT: %s", exc)
        return f"Ошибка сети при запросе к Яндекс GPT: {str(exc)}"
    except KeyError as exc:
        logger.error(
            "Неожиданный формат ответа от Yandex GPT: %s, Ответ: %s",
            exc,
            answer_text if 'response' in locals() else 'Нет ответа'
        )
        return f"Ошибка формата ответа от Яндекс GPT: {str(exc)}"
    except Exception as exc:
        logger.error("Неожиданная ошибка при запросе к Yandex GPT: %s", exc, exc_info=True)
        return f"Неизвестная ошибка при запросе к Яндекс GPT: {str(exc)}"

# Api
@app.get("/health", response_model=HealthCheckResponse, tags=["Мониторинг"], summary="Проверка состояния сервиса",
         description="Возвращает статус 'ok', если сервис работает.")
async def health_check():
    """Проверка состояния сервиса"""
    return {"status": "ok"}

@app.post("/compare_answers/", response_model=CompareAnswersResponse, tags=["Вопросы"],
          summary="Сравнить качество ответов",
          description="Сравнивает ответы локальной модели (Qwen) и Yandex GPT на заданный вопрос и оценивает их схожесть.")
async def compare_answers(request_data: QuestionRequest):
    """Сравнить качество ответов"""
    logger.info(
        "Сравнение ответов для пользователя %s с вопросом: %s",
        request_data.user_id, request_data.question
    )
    formatted_question = f"Ответь на русском языке: {request_data.question}"
    try:
        logger.info("Получение ответа от локальной LLM...")
        local_answer = llm.invoke(formatted_question)
        logger.info("Получен ответ от локальной LLM.")
    except Exception as exc:
        logger.error("Ошибка при получении ответа от локальной LLM: %s", exc)
        local_answer = f"Ошибка получения ответа от локальной модели: {str(exc)}"
    logger.info("Получение ответа от Yandex GPT...")
    yandex_answer = get_yandex_gpt_response(formatted_question)
    try:
        logger.info("Расчет сходства...")
        local_emb = np.array(embeddings.embed_query(local_answer)).reshape(1, -1)
        yandex_emb = np.array(embeddings.embed_query(yandex_answer)).reshape(1, -1)
        similarity = cosine_similarity(local_emb, yandex_emb)[0][0]
        quality_score = (similarity + 1) / 2 * 100
        logger.info(
            "Сходство рассчитано: %.4f, Оценка качества: %.2f", similarity, quality_score
        )
    except Exception as exc:
        logger.error("Ошибка при расчете сходства: %s", exc)
        similarity = 0.0
        quality_score = 0.0
    interpretation = (
        "Отлично" if quality_score > 90 else
        "Хорошо" if quality_score > 75 else
        "Удовлетворительно" if quality_score > 60 else
        "Нужно улучшить"
    )
    return CompareAnswersResponse(
        question=request_data.question,
        local_answer=local_answer,
        yandex_answer=yandex_answer,
        quality_score=float(quality_score),
        cosine_similarity_score=float(similarity),
        interpretation=interpretation
    )

@app.post("/answer_question/", response_model=AnswerQuestionResponse, tags=["Вопросы"], summary="Ответить на вопрос",
          description="Использует RetrievalQA цепочку для поиска информации в Neo4j и генерации ответа на вопрос пользователя.")
async def answer_question(request_data: QuestionRequest):
    """Ответить на вопрос"""
    logger.info(
        "Ответ на вопрос пользователя %s: %s",
        request_data.user_id, request_data.question
    )
    try:
        logger.info("Создание индекса Neo4jVector...")
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            node_label=["Document", "Entity", "Concept"],
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        logger.info("Индекс Neo4jVector создан.")
        logger.info("Создание цепочки RetrievalQA...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_index.as_retriever()
        )
        logger.info("Цепочка RetrievalQA создана.")
        query = f"Ответь на русском языке: {request_data.question}"
        logger.info("Запуск цепочки QA с запросом: %s", query)
        result = qa_chain.invoke({"query": query})
        answer = result.get("result", "Ответ не найден или ошибка в цепочке.")
        logger.info("Цепочка QA успешно завершена.")
    except ValueError as exc:
        if "Index with name vector already exists" in str(exc) and "dimensions do not match" in str(exc):
            error_msg = (
                f"Ошибка векторного индекса Neo4j: {exc}. Это часто происходит, "
                f"если модель эмбеддинга изменилась. Пожалуйста, удалите существующий "
                f"индекс 'vector' в Neo4j и повторно обработайте свои документы."
            )
            logger.error(error_msg)
            # Возвращаем 422 Unprocessable Entity для ошибок конфигурации данных
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Ошибка конфигурации векторного индекса в Neo4j: {str(exc)}. "
                       "Возможно, размерность эмбеддингов не совпадает. "
                       "Обратитесь к администратору системы."
            )
        else:
            logger.error("ValueError в answer_question: %s", exc)
            raise HTTPException(status_code=500, detail=f"Ошибка конфигурации: {str(exc)}") # from exc
    except Exception as exc:
        logger.error("Неожиданная ошибка в answer_question: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке вопроса: {str(exc)}") # from exc
    return AnswerQuestionResponse(answer=answer)

@app.get("/get_topics/", response_model=GetTopicsResponse, tags=["Справочник"], summary="Получить список тем",
         description="Возвращает список всех тем, доступных в справочнике.")
async def get_topics():
    """Получить список тем"""
    logger.info("Получение тем из Neo4j...")
    try:
        with neo4j_driver.session() as session:
            result = session.run("MATCH (t:Topic) WHERE t.name IS NOT NULL RETURN t.name AS name")
            topics = [record["name"] for record in result if record["name"] is not None]
        if not topics:
            logger.info("В базе данных не найдено тем с ненулевыми именами.")
        else:
            logger.info("Получено %s тем.", len(topics))
        return GetTopicsResponse(topics=topics)
    except Exception as exc:
        logger.error("Ошибка при получении тем из Neo4j: %s", exc, exc_info=True)
        # Возвращаем 500 для внутренних ошибок сервера
        raise HTTPException(status_code=500, detail=f"Ошибка при получении тем: {str(exc)}") # from exc

@app.get("/get_topic/{topic_name}", response_model=GetTopicResponse, tags=["Справочник"],
         summary="Получить информацию о теме", description="Возвращает описание и ссылки по указанной теме.")
async def get_topic(topic_name: str):
    """Получить информацию о теме"""
    logger.info("Получение данных для темы: %s", topic_name)
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (t:Topic {name: $name}) RETURN t.description AS description, t.links AS links",
                name=topic_name
            )
            record = result.single()
        if record:
            description = record.get("description") or "Описание не найдено."
            links = record.get("links") or []
            if not isinstance(links, list):
                links = [links] if links else []
            logger.info("Тема '%s' найдена.", topic_name)
            return GetTopicResponse(description=description, links=links)
        else:
            logger.warning("Тема '%s' не найдена.", topic_name)
            raise HTTPException(status_code=404, detail=f"Тема '{topic_name}' не найдена")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ошибка при получении темы '%s' из Neo4j: %s", topic_name, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при получении темы '{topic_name}': {str(exc)}"
        ) # from exc

@app.post("/process_pdf/", response_model=ProcessPdfResponse, tags=["Обработка документов"],
          summary="Обработать PDF документ",
          description="Принимает PDF файл, извлекает из него текст, строит граф знаний и загружает его в базу данных Neo4j.")
async def process_pdf(pdf_file: UploadFile = File(..., description="PDF файл для обработки.")):
    """Обработать PDF документ"""
    logger.info("Обработка загруженного PDF файла: %s", pdf_file.filename)
    temp_pdf_path = "temp_uploaded_file.pdf"
    try:
        logger.info("Сохранение загруженного файла...")
        with open(temp_pdf_path, "wb") as temp_file_handle:
            content = await pdf_file.read()
            temp_file_handle.write(content)
        logger.info("Файл сохранен.")
        logger.info("Загрузка PDF документа...")
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        if not documents:
            logger.warning("Из PDF не загружено документов.")
            return ProcessPdfResponse(message="PDF обработан, но не содержит извлекаемого текста.")
        logger.info("Загружено %s частей документа.", len(documents))
        logger.info("Разделение текста...")
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        split_documents = text_splitter.split_documents(documents)
        logger.info("Текст разделен на %s фрагментов.", len(split_documents))
        logger.info("Преобразование документов в графовые документы...")
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(split_documents)
        logger.info("Преобразовано в %s графовых документов.", len(graph_documents))
        logger.info("Добавление графовых документов в Neo4j...")
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        logger.info("Графовые документы успешно добавлены в Neo4j.")
        return ProcessPdfResponse(message="PDF обработан и граф загружен в Neo4j")
    except Exception as exc:
        logger.error("Ошибка при обработке PDF: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке PDF: {str(exc)}") # from exc
    finally:
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.info("Временный файл удален.")
            except OSError as exc:
                logger.warning("Не удалось удалить временный файл %s: %s", temp_pdf_path, exc)

@app.on_event("shutdown")
async def shutdown_event():
    """Завершение работы приложения"""
    logger.info("Завершение работы приложения...")
    if 'neo4j_driver' in globals():
        neo4j_driver.close()
        logger.info("Драйвер Neo4j закрыт.")
    logger.info("Завершение работы приложения завершено.")
