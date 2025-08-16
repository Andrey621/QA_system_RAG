import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Message, User
from unittest.mock import AsyncMock, patch
from src.bot.bot import dp, handle_question

@pytest.mark.asyncio
async def test_handle_question():
    message = AsyncMock(spec=Message)
    message.text = "Какими достоинствами обладают продукты на основе злаков?"
    message.from_user = User(id=123, is_bot=False, first_name="TestUser")
    message.answer = AsyncMock()

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"answer": "Продукты на основе злаков обладают рядом достоинств: они вкусны, питательны и полезны для здоровья."}
        mock_post.return_value.__aenter__.return_value = mock_response
        await handle_question(message)
        message.answer.assert_called_once_with("Продукты на основе злаков обладают рядом достоинств: они вкусны, питательны и полезны для здоровья.")