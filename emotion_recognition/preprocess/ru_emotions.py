import re
import string

import numpy as np
import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from num2words import num2words
from pandas import DataFrame
from pymorphy2 import MorphAnalyzer

from emotion_recognition.utils import EKMAN_EMOTIONS


def map_emotions_to_ekman(df: DataFrame) -> pd.DataFrame:
    """Преобразует датасет ru_go_emotions в DataFrame с эмоциями по Экману."""
    # Маппинг эмоций на эмоции по Экману
    emotion_mapping = {
        'anger': ['anger', 'annoyance', 'disapproval'],
        'disgust': ['disgust'],
        'fear': ['fear', 'nervousness', "embarrassment"],
        'joy': ['joy', 'amusement', 'excitement', 'gratitude', 'love', 'admiration',
                'approval', 'caring', 'desire', 'optimism', 'pride', 'relief'],
        'sadness': ['sadness', 'disappointment', 'grief', 'remorse'],
        'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
        'neutral': ['neutral']
    }

    df_clean = pd.DataFrame()
    df_clean["text"] = df["ru_text"]

    # Создаем колонки для эмоций по Экману
    for ekman_emotion, original_emotions in emotion_mapping.items():
        df_clean[ekman_emotion] = df[original_emotions].max(axis=1)

    # Присваиваем neutral=1, если все остальные эмоции = 0
    df_clean['neutral'] = (df_clean[EKMAN_EMOTIONS[:-1]].sum(axis=1) == 0).astype(int)
    return df_clean

def initialize_text_preprocessor():
    """Инициализирует инструменты для предобработки текста"""
    morph = MorphAnalyzer()
    download('stopwords')
    russian_stopwords = stopwords.words('russian')
    return morph, russian_stopwords


# Инициализация один раз при загрузке модуля
morph, russian_stopwords = initialize_text_preprocessor()


def preprocess_text(
        text: str,
        config: dict = {
            "remove_punct": True,
            "remove_emojis": True,
            "nums_to_words": True,
            "tokenize": True,
            "lowercase": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "remove_extra_spaces": True
        }
) -> str | list[str]:
    """
    Предобрабатывает текст с заданными параметрами

    Параметры:
        text: Исходный текст для обработки
        config: Словарь с настройками обработки:
            - remove_punct: Удалять пунктуацию (по умолчанию True)
            - remove_emojis: Удалять эмодзи (по умолчанию True)
            - nums_to_words: Преобразовывать числа в слова (по умолчанию True)
            - tokenize: Возвращать список токенов вместо строки (по умолчанию True)
            - lowercase: Приводить к нижнему регистру (по умолчанию True)
            - remove_stopwords: Удалять стоп-слова (по умолчанию True)
            - lemmatize: Лемматизировать слова (по умолчанию True)
            - remove_extra_spaces: Удалять лишние пробелы (по умолчанию True)

    Возвращает:
        Обработанный текст (str) или список токенов (list[str]) в зависимости от tokenize
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    processed_text = text

    # 1. Удаление лишних пробелов
    if config.get("remove_extra_spaces", True):
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

    # 2. Очистка пунктуации
    if config.get("remove_punct", True):
        processed_text = processed_text.translate(
            str.maketrans("", "", string.punctuation + "«»—")
        )

    # 3. Удаление эмодзи и спецсимволов
    if config.get("remove_emojis", True):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # базовые эмодзи
            u"\U0001F300-\U0001F5FF"  # символы и пиктограммы
            u"\U0001F680-\U0001F6FF"  # транспорт и карты
            u"\U0001F700-\U0001F77F"  # алхимические символы
            u"\U0001F780-\U0001F7FF"  # геометрические фигуры
            u"\U0001F800-\U0001F8FF"  # дополнительные стрелки
            u"\U0001F900-\U0001F9FF"  # дополнительные символы
            u"\U0001FA00-\U0001FA6F"  # шахматы
            u"\U0001FA70-\U0001FAFF"  # дополнительные символы и пиктограммы
            u"\U00002702-\U000027B0"  # дополнительные символы
            u"\U000024C2-\U0001F251"  # enclosed characters
            "]+",
            flags=re.UNICODE
        )
        processed_text = emoji_pattern.sub(r'', processed_text)

    # 4. Перевод цифр к словам
    if config.get("nums_to_words", True):
        words = []
        for word in processed_text.split():
            if word.isdigit():
                try:
                    word = num2words(int(word), lang='ru')
                except:
                    pass
            words.append(word)
        processed_text = " ".join(words)

    # 5. Приведение к нижнему регистру (кроме слов в CAPS)
    if config.get("lowercase", True):
        processed_text = " ".join(
            [word.lower() if not word.isupper() else word for word in processed_text.split()]
        )

    # 6. Токенизация по словам
    tokens = processed_text.split()

    # 7. Удаление стоп-слов
    if config.get("remove_stopwords", True) and config.get("tokenize", True):
        tokens = [word for word in tokens if word.lower() not in russian_stopwords]

    # 8. Лемматизация
    if config.get("lemmatize", True):
        new_tokens = []
        for token in tokens:
            t = morph.parse(token)[0].normal_form
            if token.isupper():
                t = t.upper()
            new_tokens.append(t)
        tokens = new_tokens

    return " ".join(tokens) if not config.get("tokenize", True) else tokens

def single_label(row):
    if row.sum() > 1:
        emotions = row[row == 1].index.tolist()
        chosen_emotion = np.random.choice(emotions)
        row[:] = 0
        row[chosen_emotion] = 1
    return row
