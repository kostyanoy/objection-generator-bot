import pickle
from typing import Union, List

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVectorizerWrapper:
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.is_fitted = False

    def fit(self, df: DataFrame):
        """Обучает векторайзер"""
        # Подготовка текста
        texts = self._prepare_texts(df)

        # Обучение и преобразование
        self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def transform(self, new_data: Union[DataFrame, List[str], str]) -> DataFrame:
        """Преобразует новые данные с использованием обученного векторайзера"""
        if not self.is_fitted:
            raise RuntimeError("Векторайзер не обучен. Сначала вызовите fit()")

        # Подготовка текста
        if isinstance(new_data, DataFrame):
            texts = self._prepare_texts(new_data)
        else:
            texts = [new_data] if isinstance(new_data, str) else new_data
            texts = [' '.join(t) if isinstance(t, list) else t for t in texts]

        # Преобразование
        tfidf_matrix = self.vectorizer.transform(texts)

        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

    def save(self, filepath: str) -> None:
        """Сохраняет обученный векторайзер в файл с помощью pickle"""
        if not self.is_fitted:
            raise RuntimeError("Векторайзер не обучен. Нечего сохранять.")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Модель сохранена в файл {filepath}")

    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'TfidfVectorizerWrapper':
        """Загружает векторайзер из файла и возвращает новый экземпляр класса"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Создаем новый экземпляр с переданными параметрами
        wrapper = cls(**kwargs)
        wrapper.vectorizer = data['vectorizer']
        wrapper.is_fitted = data['is_fitted']

        print(f"Модель загружена из файла {filepath}")
        return wrapper

    def _prepare_texts(self, text_series):
        """Подготавливает текст для обработки"""
        return text_series.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
