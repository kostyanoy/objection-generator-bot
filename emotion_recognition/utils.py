import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EKMAN_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']


def save_pandas(df, path):
    df.to_csv(path, index=False)
    print(f"Датафрейм сохранен в {path}")


def read_pandas(path, tokens=None, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    if tokens:
        df[tokens] = df[tokens].apply(ast.literal_eval)
    print(f"Датафрейм загружен из {path}")
    return df


def get_cached_mean_vectors(name: str, vectorizer=None, texts=None, force_reload: bool = False) -> pd.DataFrame:
    """
    Кэширует векторизованные данные с поддержкой DataFrame.

    :param vectorizer: Векторизатор (должен иметь метод transform)
    :param name: Уникальное имя для файла кэша
    :param texts: Входные тексты для векторизации
    :param force_reload: Игнорировать кэш и пересчитать данные
    :return: DataFrame с векторами
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{name}.pkl"

    # Загрузка из кэша (если не требуется перезапись)
    if not force_reload and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                print(f"Загрузка кешированных векторов из {cache_path}")
                return pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            print(f"Ошибка загрузки кэша {cache_path}: {e}. Пересчитываем...")

    print(f"Кэш не найден в {cache_path}. Векторизируем...")
    vectors = vectorizer.transform_mean(texts)

    # Конвертация в DataFrame (если ещё не)
    if not isinstance(vectors, pd.DataFrame):
        vectors = pd.DataFrame(vectors, index=texts.index if hasattr(texts, 'index') else None)

    with open(cache_path, 'wb') as f:
        pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Сохранение кешированных векторов в {cache_path}")

    return vectors

def get_cached_seq_vectors(name: str, vectorizer=None, texts=None, max_len=20, force_reload: bool = False) -> np.ndarray:
    """
    Кэширует векторизованные данные с поддержкой DataFrame.

    :param vectorizer: Векторизатор (должен иметь метод transform)
    :param name: Уникальное имя для файла кэша
    :param texts: Входные тексты для векторизации
    :param max_len: Длина последовательностей
    :param force_reload: Игнорировать кэш и пересчитать данные
    :return: DataFrame с векторами
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{name}.npy"

    # Загрузка из кэша (если не требуется перезапись)
    if not force_reload and cache_path.exists():
        try:
            print(f"Загрузка кешированных векторов из {cache_path}")
            return np.load(str(cache_path))
        except (pickle.PickleError, EOFError) as e:
            print(f"Ошибка загрузки кэша {cache_path}: {e}. Пересчитываем...")

    print(f"Кэш не найден в {cache_path}. Векторизируем...")
    vectors = vectorizer.transform_sequence(texts, max_len=max_len)

    np.save(str(cache_path), vectors)
    print(f"Сохранение кешированных векторов в {cache_path}")

    return vectors


def split_data(X, y, seed=None):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    return X_train, y_train, X_test, y_test, X_val, y_val

def combine_embeddings(X_seq, X_global, axis=-1):
    """
    Добавляет глобальный вектор к каждому токену в последовательности
    :param X_seq: [n_samples, seq_len, dim1]
    :param X_global: [n_samples, dim2]
    :return: [n_samples, seq_len, dim1 + dim2]
    """
    n_samples, seq_len, dim_seq = X_seq.shape
    _, dim_global = X_global.shape

    # Расширяем X_global до размера [n_samples, seq_len, dim_global]
    X_global_expanded = np.repeat(X_global[:, np.newaxis, :], seq_len, axis=1)

    # Конкатенируем
    return np.concatenate([X_seq, X_global_expanded], axis=axis)