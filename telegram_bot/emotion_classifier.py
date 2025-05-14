import logging
import random

from emotion_recognition.progress.model_load import process_model


class EmotionClassifier:
    _emotion_score = {
        "neutral": 0.0,
        "joy": 1.0,
        "anger": -1.0,
        "sadness": -0.5,
        "surprise": 0.2,
        "fear": -0.3,
        "disgust": -0.8
    }

    def __init__(self, vectorizer, model, max_len=20):
        self.vectorizer = vectorizer
        self.model = model
        self.max_len = max_len
        self.logger = logging.getLogger()

    def classify(self, text):
        result = process_model(self.vectorizer, self.model, self.max_len, text)
        self.logger.info(f"Для текста {text} предсказана метка {result}")
        return result

    def get_score(self, emotion):
        if emotion in self._emotion_score:
            return self._emotion_score[emotion]
        return 0.0
