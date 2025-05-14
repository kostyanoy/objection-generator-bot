import pickle
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from gensim.models import FastText


class FastTextVectorizer:
    def __init__(self, vector_size: int = 100, window: int = 5,
                 min_count: int = 3, sg: int = 1, **kwargs):
        """
        Инициализация FastText векторизатора

        :param vector_size: Размер вектора слова
        :param window: Размер окна контекста
        :param min_count: Минимальная частота слова
        :param sg: Алгоритм: 1 для skip-gram, 0 для CBOW
        :param kwargs: Другие параметры FastText
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model_params = kwargs
        self.model: Optional[FastText] = None
        self.is_trained = False

    def fit(self, df: pd.DataFrame, **kwargs):
        """Обучает векторайзер"""
        self.model = FastText(
            sentences=df.tolist(),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=4,
            epochs=10,
            **self.model_params,
            **kwargs
        )
        self.is_trained = True

    def transform_mean(self, texts: Union[pd.Series, List[List[str]]],
                       strategy: str = "mean",
                       normalize: bool = True) -> pd.DataFrame:
        """
        Преобразование текстов в векторы

        :param texts: Токенизированные тексты
        :param strategy: 'mean' или 'sum' для агрегации векторов слов
        :return: DataFrame с векторами
        """
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите train()")

        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        vectors = []
        for text in texts:
            word_vectors = []
            for word in text:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
                # Обработка OOV через субсловные единицы (если слово неизвестно)
                elif hasattr(self.model.wv, 'get_vector'):
                    try:
                        word_vectors.append(self.model.wv.get_vector(word))
                    except KeyError:
                        pass

            if len(word_vectors) > 0:
                if strategy == "mean":
                    vec = np.mean(word_vectors, axis=0)
                elif strategy == "sum":
                    vec = np.sum(word_vectors, axis=0)
                else:
                    raise ValueError("Неизвестная стратегия агрегации")

                vectors.append(vec)
            else:
                vectors.append(np.zeros(self.vector_size))

        features = pd.DataFrame(vectors)
        features.columns = [f'ft_{i}' for i in range(self.vector_size)]
        return features

    def transform_sequence(self, tokens_list: Union[pd.Series, List[List[str]]],
                           max_len: int = None,
                           padding: str = 'post',
                           truncating: str = 'post') -> np.ndarray:
        """
        Преобразование текстов в последовательности векторов для LSTM

        :param tokens_list: Список токенизированных предложений
        :param max_len: Максимальная длина последовательности (если None - берется максимальная длина в данных)
        :param padding: 'pre' или 'post' - где добавлять нулевые вектора
        :param truncating: 'pre' или 'post' - где обрезать последовательность
        :return: 3D массив (samples, timesteps, features)
        """
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")

        if isinstance(tokens_list, pd.Series):
            tokens_list = tokens_list.tolist()

        # Определяем максимальную длину
        if max_len is None:
            max_len = max(len(tokens) for tokens in tokens_list)

        sequences = []
        for tokens in tokens_list:
            sequence = []
            for token in tokens:
                # Пробуем получить вектор слова или его субсловные представления
                try:
                    vector = self.model.wv[token]
                except KeyError:
                    # Для OOV слов используем нулевой вектор
                    vector = np.zeros(self.vector_size)

                sequence.append(vector)

            # Обрезаем или дополняем последовательность
            if len(sequence) > max_len:
                if truncating == 'pre':
                    sequence = sequence[-max_len:]
                else:
                    sequence = sequence[:max_len]
            else:
                pad = [np.zeros(self.vector_size)] * (max_len - len(sequence))
                if padding == 'pre':
                    sequence = pad + sequence
                else:
                    sequence = sequence + pad

            sequences.append(sequence)

        return np.array(sequences)

    def save(self, filepath: str) -> None:
        """Сохранение модели в файл"""
        if not self.is_trained:
            raise RuntimeError("Нет обученной модели для сохранения")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vector_size': self.vector_size,
                'window': self.window,
                'min_count': self.min_count,
                'sg': self.sg,
                'model_params': self.model_params,
                'is_trained': self.is_trained
            }, f)
        print(f"Модель сохранена в файл {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FastTextVectorizer':
        """Загрузка модели из файла"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        vectorizer = cls(
            vector_size=data['vector_size'],
            window=data['window'],
            min_count=data['min_count'],
            sg=data['sg'],
            **data['model_params']
        )
        vectorizer.model = data['model']
        vectorizer.is_trained = data['is_trained']
        print(f"Модель загружена из файла {filepath}")

        return vectorizer
