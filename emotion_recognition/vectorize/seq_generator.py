import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence


class VectorizedSequenceGenerator(Sequence):
    def __init__(self, texts, labels, vectorizer, max_len, batch_size, shuffle=True):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(texts))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_texts = [self.texts[i] for i in batch_indices]

        # Векторизация на лету для каждого батча
        X = self.vectorizer.transform_sequence(batch_texts, max_len=self.max_len)
        y = self.labels[batch_indices]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)