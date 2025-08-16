import pytest
from fastapi.testclient import TestClient
from src.server.server import app

client = TestClient(app)

def test_answer_question():
    payload = {
        "user_id": 1,
        "question": "Какими достоинствами обладают продукты на основе злаков?"
    }

    response = client.post("/answer_question/", json=payload)
    assert response.status_code == 200
    assert "answer" in response.json()
    answer = response.json()["answer"]
    assert answer != ""
    assert "злак" in answer.lower() or "клетчатка" in answer.lower() or "витамин" in answer.lower()