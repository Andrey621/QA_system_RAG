""" Модели для ответов сервера """

from typing import List

from pydantic import BaseModel, Field

class HealthCheckResponse(BaseModel):
    """Модель ответа для эндпоинта проверки состояния."""
    status: str = Field(example="ok")

class QuestionRequest(BaseModel):
    """Модель запроса для вопросов."""
    user_id: int = Field(..., example=12345, description="Идентификатор пользователя Telegram.")
    question: str = Field(..., example="Каковы основные принципы работы этой системы?",
                          description="Текст вопроса пользователя.")

class CompareAnswersResponse(BaseModel):
    """Модель ответа для сравнения ответов."""
    question: str = Field(description="Вопрос пользователя.")
    local_answer: str = Field(description="Ответ, сгенерированный локальной моделью (Qwen).")
    yandex_answer: str = Field(description="Ответ, сгенерированный Yandex GPT.")
    quality_score: float = Field(ge=0.0, le=100.0, description="Нормализованная оценка качества (0-100).")
    cosine_similarity_score: float = Field(description="Косинусное сходство между эмбеддингами двух ответов.")
    interpretation: str = Field(description="Текстовая интерпретация оценки качества.")

class AnswerQuestionResponse(BaseModel):
    """Модель ответа для вопроса к ИИ."""
    answer: str = Field(example="Ответ на ваш вопрос основан на обработке документа...",
                        description="Ответ, сгенерированный системой.")

class GetTopicsResponse(BaseModel):
    """Модель ответа для получения списка тем."""
    topics: List[str] = Field(example=["Настройка системы", "API", "Устранение неполадок"],
                              description="Список названий доступных тем.")

class GetTopicResponse(BaseModel):
    """Модель ответа для получения информации о теме."""
    description: str = Field(example="Этот раздел описывает интерфейсы прикладного программирования системы.",
                             description="Описание темы.")
    links: List[str] = Field(example=["https://example.com/docs/api1", "https://example.com/docs/api2"],
                             description="Список ссылок, связанных с темой.")

class ProcessPdfResponse(BaseModel):
    """Модель ответа для обработки PDF."""
    message: str = Field(example="PDF обработан и граф загружен в Neo4j",
                         description="Сообщение о результате обработки.")