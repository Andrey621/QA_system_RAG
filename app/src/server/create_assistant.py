import os

from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk._types.expiration import ExpirationPolicy, ExpirationConfig

load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_INDEX_FILE = os.getenv("YANDEX_INDEX_FILE")  # Файлы, для которых создан индекс

sdk = YCloudML(
    folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY
)

long_ttl_config = ExpirationConfig(
    ttl_days=365 * 10, # 10 лет
    expiration_policy=ExpirationPolicy.STATIC
)

model = sdk.models.completions("yandexgpt-lite", model_version="rc")
tool = sdk.tools.search_index([YANDEX_INDEX_FILE])

INST = """
Ты работаешь с текстом на русском языке. Ты должен извлекать сущности и связи между ними, а также отвечать на вопросы на русском языке.
"""

assistant = sdk.assistants.create(model,temperature=0.3, instruction=INST, tools=[tool], expiration_policy=ExpirationPolicy.STATIC, ttl_days=365)
print(assistant)  # Вывод отсюда сохраняем в .env