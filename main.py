import os
import telebot
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Получаем токен из переменной окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Загрузка модели Qwen
model_name = "Qwen/Qwen2.5-Coder-328M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Создаем пайплайн
qa = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Инициализируем бота
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Хранилище для истории диалога (на время работы бота)
dialogue_history = {}

MAX_HISTORY = 4  # сколько пар сообщений хранить

def add_to_history(user_id, user_msg, bot_reply):
    if user_id not in dialogue_history:
        dialogue_history[user_id] = []
    history = dialogue_history[user_id]
    history.append(f"Пользователь: {user_msg}")
    history.append(f"Бот: {bot_reply}")
    # Ограничиваем длину истории
    if len(history) > MAX_HISTORY * 2:
        dialogue_history[user_id] = history[-MAX_HISTORY * 2 :]

def get_prompt_with_history(user_id, user_msg):
    prompt = "Вы — полезный помощник. Вот история вашего диалога:\n"
    if user_id in dialogue_history:
        prompt += "\n".join(dialogue_history[user_id]) + "\n"
    prompt += f"Пользователь: {user_msg}\nБот:"
    return prompt

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я всегда онлайн и помню наш разговор.")
    dialogue_history[message.chat.id] = []  # начальная инициализация

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text.strip()
    user_id = message.chat.id

    try:
        # Генерируем промпт с историей
        prompt = get_prompt_with_history(user_id, user_text)
        response = qa(
            prompt,
            max_length=150,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id
        )
        reply_text = response[0]['generated_text'].strip()

        # Отправляем ответ
        bot.reply_to(message, reply_text)

        # Сохраняем в историю
        add_to_history(user_id, user_text, reply_text)

    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

print("Бот запущен...")
bot.polling()
