import pytest
from src.server.server import driver, vector_index, qa_chain

def test_neo4j_and_qa_chain_integration():
    try:
        driver.verify_connectivity()
        neo4j_connected = True
    except Exception as e:
        neo4j_connected = False
        print(f"Neo4j connection failed: {e}")

    assert neo4j_connected is True, "Не удалось установить соединение с Neo4j."

    assert vector_index is not None, "Векторный индекс не был создан."

    query = "Какими достоинствами обладают продукты на основе злаков?"
    response = qa_chain({"query": query})

    assert "result" in response, "Ответ не содержит поле 'result'."
    assert response["result"] != "", "Ответ пустой."

    assert any(
        keyword in response["result"].lower()
        for keyword in ["злак", "клетчатка", "витамин"]
    ), "Ответ не содержит ожидаемых ключевых слов."