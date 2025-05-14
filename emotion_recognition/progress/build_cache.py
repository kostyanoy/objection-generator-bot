import time

from emotion_recognition.utils import read_pandas, get_cached_mean_vectors, get_cached_seq_vectors
from emotion_recognition.vectorize.bert import BertVectorizer
from emotion_recognition.vectorize.fasttest import FastTextVectorizer
from emotion_recognition.vectorize.word2vec import Word2VecVectorizer


def build_cache(nrows=50000, max_len=20, mean=True, seq=True):
    names = ["w2v", "ft", "bert"]
    vectorizers = [
        Word2VecVectorizer.load("models/vectorizers/w2v_vectorizer.pkl"),
        FastTextVectorizer.load("models/vectorizers/ft_vectorizer.pkl"),
        BertVectorizer.load("models/vectorizers/bert_vectorizer")
    ]
    tokens = ["tokens", "tokens", "text"]
    df_clean = read_pandas("data/ru_go_emotions_tokens.csv", "tokens", nrows=nrows)
    for name, vectorizer, token in zip(names, vectorizers, tokens):
        if mean:
            t = time.time()
            v = get_cached_mean_vectors(name=name + f"_mean_{nrows // 1000}k", vectorizer=vectorizer,
                                        texts=df_clean[token])
            print(time.time() - t)
            print(v)
            print(v.shape)
        if seq:
            t = time.time()
            v = get_cached_seq_vectors(name=name + f"_seq{max_len}_{nrows // 1000}k", vectorizer=vectorizer,
                                       texts=df_clean[token], max_len=max_len)
            if name == "bert":
                vectorizer.save_pca("models/vectorizers/pca_berk_100.joblib")
            print(time.time() - t)
            print(v)
            print(v.shape)
