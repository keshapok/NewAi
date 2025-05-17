import os
from aiogram import Bot, Dispatcher, types
import requests

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()

async def ai_response(text: str) -> str:
    resp = requests.post(
        "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        headers={"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"},
        json={"inputs": f"[INST] Ответь на русском: {text} [/INST]"}
    )
    return resp.json()[0]['generated_text'][:1500]

@dp.message()
async def handle_message(message: types.Message):
    await message.reply(await ai_response(message.text))

if __name__ == "__main__":
    from aiogram import executor
    executor.start_polling(dp)
