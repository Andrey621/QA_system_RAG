"""
Telegram бот для взаимодействия с ИИ и справочником через FastAPI
"""

# pylint: disable=line-too-long, too-few-public-methods, broad-exception-caught, invalid-name

import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from urllib.parse import quote
import aiohttp
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if not API_TOKEN:
    raise ValueError("API_TOKEN environment variable is not set.")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


class UserStates(StatesGroup):
    """Состояния для управления потоком взаимодействия с пользователем"""

    CHOOSING_MODE = State()
    ASKING_AI = State()
    USING_MANUAL_LISTING = State()
    USING_MANUAL_VIEWING = State()
    COMPARING = State()


# Константы - можно вынести в отдельный файл
MENU_OPTIONS = (
    "Привет! Выберите действие:\n"
    "1. Обратиться к нейросети\n"
    "2. Обратиться к справочнику\n"
    "3. Сравнить качество ответов"
)
MENU_RETURN_OPTIONS = (
    "Выберите действие:\n"
    "1. Обратиться к нейросети\n"
    "2. Обратиться к справочнику\n"
    "3. Сравнить качество ответов"
)
INVALID_CHOICE_MESSAGE = "Пожалуйста, выберите 1, 2 или 3."


# Функции, не относящиеся напрямую к боту - можно вынести в отдельный файл
async def _make_fastapi_request(
    method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Выполняет HTTP-запрос к бэкенду FastAPI"""
    url = f"{FASTAPI_URL}{endpoint}"
    try:
        async with aiohttp.ClientSession() as session:
            if method.upper() == "GET":
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
            if method.upper() == "POST":
                async with session.post(url, json=json_data) as response:
                    response.raise_for_status()
                    return await response.json()
            logger.error("Неподдерживаемый HTTP метод: %s", method)
            return None
    except aiohttp.ClientResponseError as exc:
        logger.error(
            "HTTP ошибка %s для %s %s: %s", exc.status, method, url, exc.message
        )
    except aiohttp.ClientError as exc:
        logger.error("Ошибка клиента для %s %s: %s", method, url, exc)
    except asyncio.TimeoutError:
        logger.error("Таймаут для %s %s", method, url)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error(
            "Неожиданная ошибка для %s %s: %s", method, url, exc, exc_info=True
        )
    return None


async def get_topics_from_neo4j() -> List[str]:
    """Получает список тем из Neo4j через FastAPI"""
    logger.info("Запрос списка тем из Neo4j через FastAPI...")
    response_data = await _make_fastapi_request("GET", "/get_topics/")
    if response_data and "topics" in response_data:
        topics = response_data["topics"]
        valid_topics = [
            topic for topic in topics if isinstance(topic, str) and topic.strip()
        ]
        logger.info("Успешно получено %s тем.", len(valid_topics))
        return valid_topics
    logger.warning("Не удалось получить темы или ключ 'topics' отсутствует в ответе.")
    return []


async def get_topic_data_from_neo4j(topic_name: str) -> Optional[Dict[str, Any]]:
    """Получает данные конкретной темы из Neo4j через FastAPI"""
    if not topic_name:
        logger.warning("Попытка получить данные темы с пустым названием.")
        return None
    encoded_name = quote(topic_name)
    endpoint = f"/get_topic/{encoded_name}"
    logger.info(
        "Запрос данных темы '%s' (кодировано: '%s')...", topic_name, encoded_name
    )
    response_data = await _make_fastapi_request("GET", endpoint)
    if response_data:
        if "error" in response_data:
            logger.info("Тема '%s' не найдена в Neo4j.", topic_name)
            return None
        logger.info("Успешно получены данные темы '%s'.", topic_name)
        return response_data
    logger.warning("Не удалось получить данные темы '%s'.", topic_name)
    return None


async def ask_ai(user_id: int, question: str) -> Optional[str]:
    """Отправляет вопрос локальной ИИ модели через FastAPI и получает ответ"""
    payload = {"user_id": user_id, "question": question}
    logger.info(
        "Отправка вопроса локальной ИИ для пользователя %s: %.50s...", user_id, question
    )
    response_data = await _make_fastapi_request("POST", "/answer_question/", payload)
    if response_data:
        answer = response_data.get("answer")
        if answer:
            logger.info("Получен ответ от локальной ИИ для пользователя %s.", user_id)
            return answer
        logger.warning(
            "Ключ 'answer' отсутствует в ответе для пользователя %s.", user_id
        )
        return "Нет ответа от сервера (ключ 'answer' отсутствует)."
    logger.error(
        "Не удалось получить ответ от локальной ИИ для пользователя %s.", user_id
    )
    return None


async def compare_answers(user_id: int, question: str) -> Optional[Dict[str, Any]]:
    """Отправляет вопрос для сравнения между моделями через FastAPI"""
    payload = {"user_id": user_id, "question": question}
    logger.info(
        "Отправка вопроса для сравнения для пользователя %s: %.50s...",
        user_id,
        question,
    )
    response_data = await _make_fastapi_request("POST", "/compare_answers/", payload)
    if response_data:
        logger.info("Получены результаты сравнения для пользователя %s.", user_id)
        return response_data
    logger.error(
        "Не удалось получить результаты сравнения для пользователя %s.", user_id
    )
    return None


# Бот - это можно разместить в одном файле, если бот простой (в этом случае можно)
@dp.message(Command("start"))
async def start_command(message: Message, state: FSMContext) -> None:
    """Обработчик команды /start"""
    logger.info("Пользователь %s запустил бота.", message.from_user.id)
    await message.answer(MENU_OPTIONS)
    await state.set_state(UserStates.CHOOSING_MODE)


@dp.message(Command("menu"))
async def menu_command(message: Message, state: FSMContext) -> None:
    """Обработчик команды /menu"""
    logger.info("Пользователь %s вернулся в главное меню.", message.from_user.id)
    await state.set_state(UserStates.CHOOSING_MODE)
    await message.answer(MENU_RETURN_OPTIONS)


@dp.message(UserStates.CHOOSING_MODE)
async def choose_mode(message: Message, state: FSMContext) -> None:
    """Обработчик выбора режима"""
    user_id = message.from_user.id
    user_choice = message.text
    logger.info("Пользователь %s выбрал режим: %s", user_id, user_choice)
    if user_choice == "1":
        await state.set_state(UserStates.ASKING_AI)
        await message.answer(
            "Вы выбрали обращение к нейросети. Задайте ваш вопрос.\n"
            "Чтобы вернуться в меню, введите /menu"
        )
    elif user_choice == "2":
        await state.set_state(UserStates.USING_MANUAL_LISTING)
        topics = await get_topics_from_neo4j()
        if topics:
            topics_list = "\n".join(topics)
            await message.answer(
                f"Вы выбрали справочник. Доступные темы:\n{topics_list}\n"
                "Выберите тему, введя её название, или введите 'назад' для возврата.\n"
                "Чтобы вернуться в меню, введите /menu"
            )
        else:
            logger.info("Для пользователя %s нет доступных тем.", user_id)
            await message.answer(
                "Справочник пуст или данные недоступны. Пожалуйста, попробуйте позже.\n"
                "Чтобы вернуться в меню, введите /menu"
            )
    elif user_choice == "3":
        await state.set_state(UserStates.COMPARING)
        await message.answer(
            "Вы выбрали сравнение качества ответов.\n"
            "Задайте вопрос, и я сравню ответы двух моделей.\n"
            "Чтобы вернуться в меню, введите /menu"
        )
    else:
        await message.answer(INVALID_CHOICE_MESSAGE)


@dp.message(UserStates.ASKING_AI)
async def handle_ai_question(message: Message, state: FSMContext) -> None:
    """Обработчик вопросов к ИИ"""
    user_id = message.from_user.id
    user_question = message.text
    if user_question.lower() == "/menu":
        await menu_command(message, state)
        return
    logger.info("Пользователь %s задал вопрос ИИ: %s", user_id, user_question)
    answer = await ask_ai(user_id, user_question)
    if answer:
        await message.answer(f"{answer}\nЧтобы вернуться в меню, введите /menu")
    else:
        await message.answer(
            "Произошла ошибка при обращении к нейросети. Пожалуйста, попробуйте позже.\n"
            "Чтобы вернуться в меню, введите /menu"
        )


@dp.message(UserStates.COMPARING)
async def handle_compare_question(message: Message, state: FSMContext) -> None:
    """Обработчик вопросов для сравнения"""
    user_id = message.from_user.id
    user_question = message.text
    if user_question.lower() == "/menu":
        await menu_command(message, state)
        return
    logger.info("Пользователь %s запросил сравнение для: %s", user_id, user_question)
    comparison_result = await compare_answers(user_id, user_question)
    if comparison_result:
        try:
            question_text = comparison_result.get("question", "Вопрос не указан")
            local_answer = comparison_result.get(
                "local_answer", "Ответ локальной модели не получен"
            )
            yandex_answer = comparison_result.get(
                "yandex_answer", "Ответ Yandex GPT не получен"
            )
            quality_score = comparison_result.get("quality_score", 0.0)
            cosine_similarity = comparison_result.get("cosine_similarity", 0.0)
            interpretation = comparison_result.get("interpretation", "Неизвестно")
            answer_text = (
                f"<b>Вопрос:</b> {question_text}\n"
                f"<b>Ответ локальной модели (Qwen):</b>\n{local_answer}\n"
                f"<b>Ответ Yandex GPT:</b>\n{yandex_answer}\n"
                f"<b>Оценка качества (0-100):</b> {quality_score:.2f}\n"
                f"<b>Косинусное сходство:</b> {cosine_similarity:.4f}\n"
                f"<b>Интерпретация:</b> {interpretation}"
            )
            await message.answer(answer_text, parse_mode="HTML")
            await message.answer(
                "Чтобы задать другой вопрос для сравнения или вернуться в меню, введите /menu"
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "Ошибка форматирования/отправки результата сравнения для пользователя %s: %s",
                user_id,
                exc,
                exc_info=True,
            )
            await message.answer(
                "Произошла ошибка при форматировании результата сравнения.\n"
                "Чтобы вернуться в меню, введите /menu"
            )
    else:
        await message.answer(
            "Произошла ошибка при сравнении ответов. Пожалуйста, попробуйте позже.\n"
            "Чтобы вернуться в меню, введите /menu"
        )


@dp.message(UserStates.USING_MANUAL_LISTING)
async def handle_manual_listing(message: Message, state: FSMContext) -> None:
    """Обработчик просмотра списка тем справочника"""
    user_id = message.from_user.id
    user_input = message.text
    if user_input.lower() == "/menu":
        await menu_command(message, state)
        return
    if user_input.lower() == "назад":
        logger.info("Пользователь %s вернулся из списка тем.", user_id)
        await state.set_state(UserStates.CHOOSING_MODE)
        await message.answer(MENU_RETURN_OPTIONS)
        return
    selected_topic = user_input
    logger.info("Пользователь %s выбрал тему: %s", user_id, selected_topic)
    topic_data = await get_topic_data_from_neo4j(selected_topic)
    if topic_data:
        description = topic_data.get("description", "Описание не найдено.")
        links = topic_data.get("links", [])
        links_str = "\n".join(links) if links else "Ссылки не найдены."
        await message.answer(
            f"<b>Тема:</b> {selected_topic}\n"
            f"<b>Описание:</b> {description}\n"
            f"<b>Ссылки:</b>\n{links_str}\n"
            "Введите 'назад' для возврата к списку тем.\n"
            "Чтобы вернуться в меню, введите /menu",
            parse_mode="HTML",
        )
        await state.set_state(UserStates.USING_MANUAL_VIEWING)
    else:
        logger.info(
            "Тема '%s' не найдена для пользователя %s. Обновление списка.",
            selected_topic,
            user_id,
        )
        topics = await get_topics_from_neo4j()
        if topics:
            topics_list = "\n".join(topics)
            await message.answer(
                f"Тема '{selected_topic}' не найдена. Пожалуйста, выберите из списка.\n"
                f"Доступные темы:\n{topics_list}\n"
                "Чтобы вернуться в меню, введите /menu"
            )
        else:
            logger.info(
                "Нет доступных тем для обновления списка для пользователя %s.", user_id
            )
            await message.answer(
                "Справочник пуст или данные недоступны.\n"
                "Чтобы вернуться в меню, введите /menu"
            )


@dp.message(UserStates.USING_MANUAL_VIEWING)
async def handle_manual_viewing(message: Message, state: FSMContext) -> None:
    """Обработчик просмотра деталей темы"""
    user_id = message.from_user.id
    user_input = message.text
    if user_input.lower() == "/menu":
        await menu_command(message, state)
        return
    if user_input.lower() == "назад":
        logger.info("Пользователь %s вернулся к списку тем из просмотра.", user_id)
        await state.set_state(UserStates.USING_MANUAL_LISTING)
        topics = await get_topics_from_neo4j()
        if topics:
            topics_list = "\n".join(topics)
            await message.answer(
                f"Доступные темы:\n{topics_list}\n"
                "Выберите тему, введя её название, или введите 'назад' для возврата.\n"
                "Чтобы вернуться в меню, введите /menu"
            )
        else:
            logger.info(
                "Нет доступных тем для отображения списка для пользователя %s.", user_id
            )
            await message.answer(
                "Справочник пуст или данные недоступны.\n"
                "Чтобы вернуться в меню, введите /menu"
            )
        return
    await message.answer(
        "Пожалуйста, введите 'назад' для возврата к списку тем.\n"
        "Чтобы вернуться в меню, введите /menu"
    )


async def main() -> None:
    """Точка входа в приложение"""
    logger.info("Запуск Telegram бота...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)
    logger.info("Telegram бот остановлен.")


if __name__ == "__main__":
    asyncio.run(main())
