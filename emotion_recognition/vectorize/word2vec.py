import pickle
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


class Word2VecVectorizer:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, **kwargs):
        """
        Инициализация Word2Vec векторизатора

        :param vector_size: Размер вектора слова
        :param window: Размер окна контекста
        :param min_count: Минимальная частота слова для учета
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model_params = kwargs
        self.model: Optional[Word2Vec] = None
        self.is_trained = False

    def fit(self, df: pd.DataFrame):
        """Обучает векторайзер"""
        self.model = Word2Vec(
            sentences=df.tolist(),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            **self.model_params,
        )
        self.is_trained = True

    def transform_mean(self, tokens_list: Union[pd.Series, List[List[str]]]) -> pd.DataFrame:
        """
        Преобразование текстов в векторы
        """
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit()")

        if isinstance(tokens_list, pd.Series):
            tokens_list = tokens_list.tolist()

        vectors = []
        for tokens in tokens_list:
            word_vectors = []
            for token in tokens:
                if token in self.model.wv:
                    word_vectors.append(self.model.wv[token])

            if len(word_vectors) > 0:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))

        features = pd.DataFrame(vectors)
        features.columns = [f'w2v_{i}' for i in range(self.vector_size)]
        return features

    def transform_sequence(self, tokens_list: Union[pd.Series, List[List[str]]], max_len: int = None,
                           padding: str = 'post', truncating: str = 'post') -> np.ndarray:
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
                if token in self.model.wv:
                    sequence.append(self.model.wv[token])
                else:
                    sequence.append(np.zeros(self.vector_size))

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
                'model_params': self.model_params,
                'is_trained': self.is_trained
            }, f)
        print(f"Модель сохранена в файл {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Word2VecVectorizer':
        """Загрузка модели из файла"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        vectorizer = cls(
            vector_size=data['vector_size'],
            window=data['window'],
            min_count=data['min_count'],
            **data['model_params']
        )
        vectorizer.model = data['model']
        vectorizer.is_trained = data['is_trained']
        print(f"Модель загружена из файла {filepath}")
        return vectorizer
