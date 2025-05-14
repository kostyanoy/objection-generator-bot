import torch

from emotion_recognition.architecture.transformer import load_transformer_model
from emotion_recognition.preprocess.ru_emotions import preprocess_text
from emotion_recognition.utils import EKMAN_EMOTIONS
from emotion_recognition.vectorize.fasttest import FastTextVectorizer


def get_model(path):
    input_dim = 100
    num_classes = 7

    model = load_transformer_model(path, input_dim, num_classes, device=device)
    return model


def get_vectorizer(path):
    vectorizer = FastTextVectorizer.load(path)
    return vectorizer


def process_model(vectorizer, model, max_len, text):
    tokens = preprocess_text(text)
    vector = vectorizer.transform_sequence([tokens], max_len)
    with torch.no_grad():
        vector = torch.tensor(vector).float().to(device)
        output = model(vector)
        predicted_class = torch.argmax(output, dim=1).item()
    return EKMAN_EMOTIONS[predicted_class]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
