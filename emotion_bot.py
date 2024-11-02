import os
import numpy as np
from telegram import Update
from tensorflow.keras.models import load_model
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext, Application
import logging
import asyncio
import librosa


emotional_testing_model = load_model('emotional_testing.keras')
CLASS_LIST = ['neutral', 'disgust', 'sad', 'happy', 'surprise', 'angry', 'fear']
class_list_ru = ['нейтральный', 'отвращение', 'грустный', 'счастливый', 'удивление', 'гнев', 'страх']
N_FFT = 8192                              # Размер окна преобразования Фурье для расчета спектра
HOP_LENGTH = 512                          # Объем данных для расчета одного набора признаков


# Функция параметризации аудио
def get_features(y,                     # волновое представление сигнала
                 sr,                    # частота дискретизации сигнала y
                 n_fft=N_FFT,           # размер скользящего окна БПФ
                 hop_length=HOP_LENGTH  # шаг скользящего окна БПФ
                 ):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)     # Хромаграмма
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)                   # Мел-кепстральные коэффициенты
    rmse = librosa.feature.rms(y=y, hop_length=hop_length)                                        # Среднеквадратическая амплитуда
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length) # Спектральный центроид
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)  # Ширина полосы частот
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)    # Спектральный спад частоты
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)                            # Пересечения нуля

    # Сборка признаков в общий список:
    features = {'rmse': rmse,
                'spct': spec_cent,
                'spbw': spec_bw,
                'roff': rolloff,
                'zcr' : zcr,
                'mfcc': mfcc,
                'stft': chroma_stft}

    return features

def prepare_prediction_batch(audio_file_path, duration_sec=2):
    y, sr = librosa.load(audio_file_path, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Подготовка списка для хранения каждого 2-секундного отрезка
    segments = []
    
    # Разбиваем аудио на отрезки по `duration_sec` секунд
    for start in range(0, int(total_duration), duration_sec):
        end = start + duration_sec
        segment = y[start * sr : end * sr]
        
        # Если последний отрезок короче 1 секунды и не единственный, пропускаем его
        if len(segment) < sr and start > 0:
            continue
        
        # Если единственный отрезок короче 2 секунд, дополняем его до нужной длины
        if len(segment) < duration_sec * sr:
            repeats = int(np.ceil((duration_sec * sr) / len(segment)))
            segment = np.tile(segment, repeats)[:duration_sec * sr]
        
        # Получаем признаки для сегмента и добавляем в batch
        features = get_features(segment, sr)
        feature_vector = np.hstack([f.mean(axis=1) for f in features.values()])  # Конкатенируем признаки
        segments.append(feature_vector)
    
    # Возвращаем пакет признаков для всех сегментов в формате numpy массива
    return np.array(segments)

def format_time(seconds):
    """Форматирование времени в минуты и секунды."""
    if seconds < 60:
        return f"{seconds} сек"
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes} мин {seconds} сек"

def predict_emotion(prediction_batch, model, class_list, return_list=False):
    # Предсказания модели по каждому отрезку
    predictions = model.predict(prediction_batch)
    
    if return_list:
        prediction_with_times = []
        for i, prediction in enumerate(predictions):
            # Получаем название эмоции с максимальной вероятностью
            emotion = class_list[np.argmax(prediction)]
            
            # Вычисляем временной промежуток для текущего отрезка
            start_time = i * 2
            end_time = start_time + 2
            time_interval = f"{format_time(start_time)} - {format_time(end_time)}"
            
            # Добавляем в список результат в формате "название эмоции X-Y сек"
            prediction_with_times.append(f"{emotion} {time_interval}")
        
        return prediction_with_times
    
    # Если требуется одно итоговое предсказание
    final_prediction = np.sum(predictions, axis=0)
    return class_list[np.argmax(final_prediction)]


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
    prediction_batch = prepare_prediction_batch(file_path, duration_sec=2)
    emotion = predict_emotion(prediction_batch, emotional_testing_model, class_list=class_list_ru, return_list=False)
    
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
