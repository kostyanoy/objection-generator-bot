from emotion_recognition.utils import read_pandas
from emotion_recognition.vectorize.bert import BertVectorizer
from emotion_recognition.vectorize.fasttest import FastTextVectorizer
from emotion_recognition.vectorize.tfidf import TfidfVectorizerWrapper
from emotion_recognition.vectorize.word2vec import Word2VecVectorizer


def build_tfidf():
    df_clean = read_pandas("data/ru_go_emotions_tokens.csv", tokens="tokens")
    tfidf = TfidfVectorizerWrapper()
    tfidf.fit(df_clean["tokens"])
    tfidf.save("models/tfidf_vectorizer.pkl")


def build_w2v():
    df_clean = read_pandas("data/ru_go_emotions_tokens.csv", tokens="tokens")
    w2v = Word2VecVectorizer()
    w2v.fit(df_clean["tokens"])
    w2v.save("models/w2v_vectorizer.pkl")


def build_ft():
    df_clean = read_pandas("data/ru_go_emotions_tokens.csv", tokens="tokens")
    ft = FastTextVectorizer()
    ft.fit(df_clean["tokens"])
    ft.save("models/ft_vectorizer.pkl")


def build_bert():
    bert = BertVectorizer()
    bert.save("models/bert_vectorizer")