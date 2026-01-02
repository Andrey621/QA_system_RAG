""" Модели для ответов сервера """

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Модель ответа для эндпоинта проверки состояния"""
    status: str = Field(example="ok")

class QuestionRequest(BaseModel):
    """Модель запроса для вопросов"""
    user_id: int = Field(..., example=12345, description="Идентификатор пользователя Telegram.")
    question: str = Field(..., example="Каковы основные принципы работы этой системы?",
                          description="Текст вопроса пользователя.")

class CompareAnswersResponse(BaseModel):
    """Модель ответа для сравнения ответов"""
    question: str = Field(description="Вопрос пользователя.")
    local_answer: str = Field(description="Ответ, сгенерированный локальной моделью (Qwen).")
    yandex_answer: str = Field(description="Ответ, сгенерированный Yandex GPT.")
    quality_score: float = Field(ge=0.0, le=100.0, description="Нормализованная оценка качества (0-100).")
    cosine_similarity_score: float = Field(description="Косинусное сходство между эмбеддингами двух ответов.")
    interpretation: str = Field(description="Текстовая интерпретация оценки качества.")

class AnswerQuestionResponse(BaseModel):
    """Модель ответа для вопроса к ИИ"""    
    answer: str = Field(example="Ответ на ваш вопрос основан на обработке документа...",
                        description="Ответ, сгенерированный системой.")

class GetTopicsResponse(BaseModel):
    """Модель ответа для получения списка тем"""
    topics: List[str] = Field(example=["Настройка системы", "API", "Устранение неполадок"],
                              description="Список названий доступных тем.")

class GetTopicResponse(BaseModel):
    """Модель ответа для получения информации о теме"""
    description: str = Field(example="Этот раздел описывает интерфейсы прикладного программирования системы.",
                             description="Описание темы.")
    links: List[str] = Field(example=["https://example.com/docs/api1", "https://example.com/docs/api2"],
                             description="Список ссылок, связанных с темой.")

class ProcessPdfResponse(BaseModel):
    """Модель ответа для обработки PDF"""
    message: str = Field(example="PDF обработан и граф загружен в Neo4j",
                         description="Сообщение о результате обработки.")


class FineTuneExample(BaseModel):
    """Пример для дообучения"""
    instruction: str = Field(..., example="Каковы основные принципы работы этой системы?",
                             description="Инструкция")
    response: str = Field(..., example="Ответ на ваш вопрос основан на обработке документа...",
                          description="Ответ")
    context: Optional[str] = Field(default=None, example="Контекст", description="Контекст")
    task: Optional[str] = Field(default=None, description="Тип задачи")
    labels: Optional[List[str]] = Field(default=None, description="Метки")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Дополнительные метаданные")

class FineTuneRequest(BaseModel):
    """Запрос на дообучение"""
    user_id: int = Field(..., example=12345, description="Telegram user identifier.")
    examples: List[FineTuneExample] = Field(..., description="Примеры для дообучения")
    base_model_name: Optional[str] = Field(default="Qwen/Qwen1.5-0.5B-Chat",
                                           description="Имя базовой модели HuggingFace")
    output_dir: Optional[str] = Field(default=None, description="Путь к директории для сохранения дообученной модели")
    learning_rate: float = Field(default=5e-5, gt=0.0, description="Скорость обучения")
    num_epochs: int = Field(default=1, ge=1, description="Количество эпох")
    max_seq_length: int = Field(default=512, ge=128, le=4096, description="Максимальная длина последовательности")
    gradient_accumulation_steps: int = Field(default=1, ge=1,
                                             description="Количество шагов аккумуляции градиента")
    run_name: Optional[str] = Field(default=None, description="Имя запуска")
    trust_remote_code: bool = Field(default=True, description="Доверять удаленному коду")

class FineTuneResponse(BaseModel):
    """Ответ на запрос на дообучение"""
    status: str = Field(example="completed", description="Статус дообучения")
    run_name: str = Field(example="user-12345-20240101-120000", description="Имя запуска")
    model_path: str = Field(example="/app/models/fine_tuned/user-12345-20240101-120000",
                            description="Путь к обученной модели")
    message: str = Field(example="Дообучение завершено.", description="Сообщение о результате")

class FineTuneJobResponse(BaseModel):
    """Ответ на запрос на дообучение"""
    job_id: str = Field(example="job-b1d7aa5c", description="Идентификатор дообучения")
    status: str = Field(example="queued", description="Текущий статус")
    message: str = Field(example="Fine-tune job enqueued.", description="Сообщение о статусе")

class FineTuneJobStatusResponse(BaseModel):
    """Статус дообучения"""
    job_id: str = Field(example="job-b1d7aa5c", description="Идентификатор дообучения")
    status: str = Field(example="running", description="Текущий статус")
    message: Optional[str] = Field(default=None, description="Сообщение")
    model_path: Optional[str] = Field(default=None, description="Путь к обученной модели")
