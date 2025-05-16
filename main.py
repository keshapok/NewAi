import os
import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B"  # можно выбрать другую из https://huggingface.co/models 

# Адрес API
API_URL = f"https://api-inference.huggingface.co/models/ {MODEL_NAME}"

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Задавайте свой вопрос — я постараюсь ответить.")

# Обработчик сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    await update.message.reply_text("Думаю над ответом...")

    try:
        output = query({
            "inputs": user_input,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.95
            }
        })

        if isinstance(output, list) and 'generated_text' in output[0]:
            answer = output[0]['generated_text']
        else:
            answer = "Не удалось получить ответ. Попробуйте снова."

        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")

# Основная функция
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await app.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
