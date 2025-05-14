import os
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
import torch
from pandas import Series
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel


class BertVectorizer:
    def __init__(self, model_name: str = 'cointegrated/rubert-tiny', device: str = 'cpu'):
        """
        :param model_name: Название модели с HuggingFace
        :param device: 'cpu' или 'cuda'
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()
        self.pca = None

    def vectorize(self, texts: Union[str, List[str]],
                  pooling: str = 'mean',
                  max_len=512,
                  batch_size: int = 1024,
                  normalize: bool = True) -> np.ndarray:
        """
        Векторизация текстов

        :param texts: Текст или список текстов
        :param pooling: 'mean', 'cls' или 'max' - стратегия объединения токенов
        :param batch_size размер батча
        :return: Массив эмбеддингов [n_samples, hidden_size]
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            print(f"batch {i // batch_size}")
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Выбор стратегии пулинга
            if pooling == 'cls':
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]-токен
            elif pooling == 'mean':
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            elif pooling == 'max':
                embeddings = torch.max(outputs.last_hidden_state, dim=1).values
            else:
                embeddings = outputs.last_hidden_state

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def transform_mean(self, df: pd.Series) -> pd.DataFrame:
        vectors = self.vectorize(df.tolist())
        return pd.DataFrame(vectors,
                            columns=[f'bert_{i}' for i in range(vectors.shape[1])],
                            index=df.index)

    def transform_sequence(self, df: Series, max_len: int = None, reduce_dim: int = 100) -> np.ndarray:
        """
        Преобразование текстов в последовательности векторов токенов для LSTM

        :param df: Pandas Series с текстами
        :param max_len: Максимальная длина последовательности
        :param reduce_dim: Целевая размерность после понижения
        :return: 3D массив (samples, timesteps, reduce_dim)
        """
        # Получаем последовательности токенов без пулинга (shape: [n_samples, seq_len, hidden_size])
        vectors = self.vectorize(df.tolist(), pooling="none", max_len=max_len)

        # Инициализируем PCA
        n_samples, seq_len, hidden_dim = vectors.shape
        pca = PCA(n_components=reduce_dim)

        # Сжимаем все токены в один массив
        flat_vectors = vectors.reshape(-1, hidden_dim)
        reduced_vectors = pca.fit_transform(flat_vectors)
        self.pca = pca

        # Восстанавливаем структуру последовательности
        reduced_sequences = reduced_vectors.reshape(n_samples, seq_len, reduce_dim)
        return reduced_sequences

    def save_pca(self, filepath: str):
        """Сохранение обученного PCA"""
        if not hasattr(self, 'pca') or self.pca is None:
            raise RuntimeError("Нет обученного PCA для сохранения")

        # Сохраняем через joblib
        joblib.dump(self.pca, filepath)
        print(f"PCA модель сохранена в файл {filepath}")

    def load_pca(self, filepath: str):
        """Загрузка обученного PCA из файла"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл PCA не найден: {filepath}")

        self.pca = joblib.load(filepath)
        print(f"PCA модель загружена из файла {filepath}")

    def save(self, filepath: str):
        """Сохранение модели"""
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        print(f"Модель сохранена в файл {filepath}")

    @classmethod
    def load(cls, filepath: str, **kwargs):
        """Загрузка модели"""
        print(f"Модель загружена из файла {filepath}")
        return cls(model_name=filepath, **kwargs)
