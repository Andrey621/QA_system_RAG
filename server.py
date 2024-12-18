from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import GraphDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import os

app = FastAPI()

# Устанавливаем переменные окружения для подключения к Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Инициализация драйвера Neo4j
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()  # Проверяем подключение к Neo4j
    print("Neo4j connection successful!")
except Exception as e:
    print(f"Neo4j connection failed: {e}")
    raise ValueError("Could not connect to Neo4j database. Please check the connection settings.")

# Инициализация OllamaLLM
llm = OllamaLLM(
    model="llama3:8b",
    base_url="http://localhost:11434"
)

# Загрузка модели эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Создание векторного индекса на основе существующего графа
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    search_type="hybrid",
    node_label=["Document", "Entity", "Concept"],
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Создание цепочки RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vector_index.as_retriever()
)


# Класс модели запроса
class QuestionRequest(BaseModel):
    user_id: int
    question: str


# Основной обработчик вопроса
@app.post("/answer_question/")
async def answer_question(data: QuestionRequest):
    try:
        # Добавляем указание на русский язык в запрос
        query = f"Ответь на русском языке: {data.question}"

        # Используем цепочку RetrievalQA для ответа на вопрос
        result = qa_chain({"query": query})
        answer = result["result"]
    except Exception as e:
        answer = f"Ошибка при обработке вопроса: {str(e)}"

    return {"answer": answer}