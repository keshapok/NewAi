import os
import telebot
from ctranslate2 import Generator
from transformers import AutoTokenizer

# Токен Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Путь к модели (укажи свой)
MODEL_PATH = "models/pygmalion-350m-ru-tiny"

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("TortugaAg/pygmalion-350m-ru-tiny")
model = Generator(MODEL_PATH, device="cpu", inter_threads=4)

# Хранилище истории
dialogue_history = {}

def generate_response(prompt):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
    results = model.generate_batch([tokens], max_length=150, sampling_topk=10)
    reply = tokenizer.decode(results[0].sequences_ids[0])
    return reply.strip()

def get_prompt_with_history(user_id, user_msg):
    prompt = "Вы — полезный помощник. Вот история вашего диалога:\n"
    if user_id in dialogue_history:
        prompt += "\n".join(dialogue_history[user_id]) + "\n"
    prompt += f"Пользователь: {user_msg}\nБот:"
    return prompt

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я всегда онлайн и помню наш разговор.")
    dialogue_history[message.chat.id] = []

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text.strip()
    user_id = message.chat.id

    try:
        prompt = get_prompt_with_history(user_id, user_text)
        reply = generate_response(prompt)
        bot.reply_to(message, reply)
        # Сохраняем историю
        if user_id not in dialogue_history:
            dialogue_history[user_id] = []
        history = dialogue_history[user_id]
        history.append(f"Пользователь: {user_text}")
        history.append(f"Бот: {reply}")
        if len(history) > 6:
            history.pop(0)
            history.pop(0)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

print("Бот запущен...")
bot.polling()
