import os
import numpy as np
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext, Application
import logging
import asyncio

# Загрузка модели и список классов
# emotional_testing_model = load_model('emotional_testing.keras')
CLASS_LIST = ['neutral', 'disgust', 'sad', 'happy', 'surprise', 'angry', 'fear']
class_list_ru = ['нейтральный', 'отвращение', 'грустный', 'счастливый', 'удивление', 'гнев', 'страх']

# Настройка логгирования
logging.basicConfig(level=logging.INFO)

# Асинхронная функция для обработки голосового сообщения
async def handle_voice(update: Update, context: CallbackContext):
    # Скачивание аудиофайла
    voice = await update.message.voice.get_file()
    
    # Создаем папку downloads, если она не существует
    os.makedirs("downloads", exist_ok=True)
    file_path = f"downloads/{voice.file_id}.ogg"
    
    # Скачиваем файл
    await voice.download_to_drive(file_path)  # Используем download_to_drive
    
    # # Обработка аудиофайла для предсказания эмоции
    # prediction_batch = await prepare_prediction_batch(file_path, duration_sec=2)
    # emotion = await predict_emotion(prediction_batch, emotional_testing_model, class_list=class_list_ru, return_list=False)
    emotion = "вы прислали гс я его скачал и удалил"
    
    # Отправка ответа с предсказанной эмоцией
    await update.message.reply_text(f"Предсказанная эмоция: {emotion}")
    
    # Удаление временных файлов
    os.remove(file_path)

# Асинхронная функция для команды /start
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Отправьте мне голосовое сообщение, и я попробую определить эмоцию.")

# Инициализация и запуск бота
def main():
    # Создаем и запускаем бота
    dp = Application.builder().token(TOKEN).build()
    
    # Обработчики команд
    dp.add_handler(CommandHandler("start", start))
    
    # Обработчик голосовых сообщений
    dp.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    # Запуск бота
    dp.run_polling()

if __name__ == '__main__':
    main()
