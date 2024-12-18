from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
import os

# Устанавливаем переменные окружения для подключения к Neo4j
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

# Инициализация Neo4jGraph
graph = Neo4jGraph()

# Инициализация OllamaLLM
llm = OllamaLLM(model="llama3:8b",
                base_url="http://localhost:11434",
                system="Ты работаешь с текстом на русском языке. Ты должен извлекать сущности и связи между ними, а также отвечать на вопросы на русском языке.")

# Загрузка и обработка PDF-файла
from langchain.document_loaders import PyPDFLoader

# Укажите путь к вашему PDF-файлу
pdf_path = "test.pdf"

# Загрузка PDF-файла
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Разбиваем текст на чанки
from langchain_text_splitters import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(documents)

# Преобразование текста в граф
from langchain_experimental.graph_transformers import LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Сохранение графа в Neo4j
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name  = "BAAI/bge-base-en-v1.5")

from langchain_community.vectorstores import Neo4jVector
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label=["Document", "Entity", "Concept"],
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
