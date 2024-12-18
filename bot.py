from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
import aiohttp
import asyncio

API_TOKEN = ""
FASTAPI_URL = "http://localhost:8000/answer_question/"

# Инициализация бота
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Задай мне любой вопрос, и я постараюсь на него ответить.")

@dp.message()
async def handle_question(message: Message):
    payload = {"user_id": message.from_user.id, "question": message.text}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(FASTAPI_URL, json=payload) as response:
                if response.status == 200:
                    response_data = await response.json()
                    answer = response_data.get("answer", "Нет ответа от сервера.")
                    await message.answer(answer)
                else:
                    await message.answer("Ошибка на стороне сервера FastAPI.")
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())