import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, filters, ApplicationBuilder, ContextTypes, CallbackQueryHandler

from emotion_recognition.progress.model_load import get_vectorizer, get_model
from telegram_bot.emotion_classifier import EmotionClassifier
from telegram_bot.video_generator import VideoGenerator

# Настройка логгера
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)  # логгер
emotion_classifier = EmotionClassifier(
    vectorizer=get_vectorizer(),
    model=get_model()
)
forwarded_messages_buffer: dict[int, VideoGenerator] = {}  # обработчики сообщений


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет! Перешли мне несколько сообщений (в том числе от разных людей), "
             "я определю эмоции и сделаю видео в стиле Ace Attorney!\n\n"
             "Используй /help чтобы узнать больше."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    TEXT_HELP = ("Как использовать:\n"
                 "1. Выбери несколько сообщений (можно от разных людей).\n"
                 "2. Нажми «Переслать» и отправь их мне.\n"
                 "3. Я сделаю видео с учётом эмоций.\n"
                 "Видео создаётся не моментально — подожди немного.\n"
                 "Если ты уже в очереди, я сообщу тебе об этом.")
    TEXT_ABOUT = "Этот бот анализирует пересланные сообщения и делает видео в стиле Ace Attorney."
    TEXT_STATUS_PROCESSING = "Ваши сообщения находятся в очереди на обработку."
    TEXT_STATUS_IDLE = "У вас нет сообщений в очереди на обработку."

    # кнопки для пользователя
    keyboard = [
        [InlineKeyboardButton("ℹ️ О боте", callback_data='about')],
        [InlineKeyboardButton("📝 Пример использования", callback_data='example')],
        [InlineKeyboardButton("📊 Статус", callback_data='status')],
        [InlineKeyboardButton("❌ Закрыть", callback_data='close')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query is None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=TEXT_HELP, reply_markup=reply_markup)
        return

    query = update.callback_query
    await query.answer()

    # ответить на нажатие по кнопке
    if query.data == 'about':
        await query.edit_message_text(text=TEXT_ABOUT, reply_markup=reply_markup)
    elif query.data == 'example':
        await query.edit_message_text(text=TEXT_HELP, reply_markup=reply_markup)
    elif query.data == 'status':
        user_id = update.effective_user.id
        if user_id in forwarded_messages_buffer and forwarded_messages_buffer[user_id].is_processing:
            await query.edit_message_text(text=TEXT_STATUS_PROCESSING, reply_markup=reply_markup)
        else:
            await query.edit_message_text(text=TEXT_STATUS_IDLE, reply_markup=reply_markup)
    elif query.data == 'close':
        await query.message.delete()


async def handle_forwarded_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Проверяем, есть ли уже запрос пользователя в обработке
    user_id = update.effective_user.id
    if user_id in forwarded_messages_buffer and forwarded_messages_buffer[user_id].is_processing:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Ожидайте, ваши сообщения уже обрабатываются. (Пересланные сообщения отклонены)")
        return

    # Собираем сообщения вместе
    if user_id not in forwarded_messages_buffer:
        chat_id = update.effective_chat.id
        forwarded_messages_buffer[user_id] = VideoGenerator(asyncio.get_event_loop(), context.bot, chat_id,
                                                            emotion_classifier)
    forwarded_messages_buffer[user_id].add_message(update)


def main():
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле")

    application = ApplicationBuilder() \
        .token(TOKEN) \
        .build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    # application.add_handler(CommandHandler("status", status))

    application.add_handler(CallbackQueryHandler(help_command))

    application.add_handler(MessageHandler(filters.FORWARDED & filters.TEXT, handle_forwarded_messages))

    # Запуск
    application.run_polling()


if __name__ == '__main__':
    main()
