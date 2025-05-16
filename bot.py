import os
import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import nest_asyncio

nest_asyncio.apply()

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "YOUR_HF_READ_TOKEN")

# Настройки модели
MODEL_NAME = "ai-forever/ruGPT-3-small"
API_URL = f"https://api-inference.huggingface.co/models/ {MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_model(question: str) -> str:
    payload = {
        "inputs": question.strip(),
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            logging.error(f"Ошибка HuggingFace: {response.status_code}, {response.text}")
            return "Не удалось получить ответ. Попробуйте позже."
    except Exception as e:
        logging.exception("Ошибка при обращении к модели")
        return f"Ошибка: {str(e)}"

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я ваш ИИ-ассистент на русском языке.")

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    logging.info(f"Пользователь: {user_input}")
    answer = query_model(user_input)
    await update.message.reply_text(answer)

# Основной запуск
if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)
