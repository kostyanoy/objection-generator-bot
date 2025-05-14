from pathlib import Path

import torch

from emotion_recognition.architecture.transformer import load_transformer_model
from emotion_recognition.preprocess.ru_emotions import preprocess_text
from emotion_recognition.utils import EKMAN_EMOTIONS
from emotion_recognition.vectorize.fasttest import FastTextVectorizer


def get_model():
    project_root = Path(__file__).parent.parent.parent
    path = project_root / "emotion_recognition" / "models" / "transformer" / "transformer_fasttext.pt"

    input_dim = 100
    num_classes = 7
    print(str(path))
    model = load_transformer_model(str(path), input_dim, num_classes, device=device)
    return model


def get_vectorizer():
    project_root = Path(__file__).parent.parent.parent
    path = project_root / "emotion_recognition" / "models" / "vectorizers" / "ft_vectorizer.pkl"
    print(str(path))
    vectorizer = FastTextVectorizer.load(str(path))
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
