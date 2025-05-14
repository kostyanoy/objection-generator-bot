import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransferedClassifier:
    def __init__(self, model_path="DeepPavlov/rubert-base-cased", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Загрузка модели с {model_path}, используем устройство: {self.device}")

        # Загружаем модель и токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=7,
            ignore_mismatched_sizes=True,
            output_hidden_states=False
        ).to(self.device)

        # ❄️ Замораживаем всё, кроме нового классификатора
        for param in self.model.parameters():
            param.requires_grad = False

        # 🔁 Новый классификатор
        in_features = self.model.config.hidden_size
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features, 7)
        ).to(self.device)

        # 🔥 Разморозим только классификатор для обучения
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=16, learning_rate=2e-5):
        train_dataset = EmotionDataset(X_train, y_train, self.tokenizer)

        eval_dataset = None
        if X_val is not None and y_val is not None:
            eval_dataset = EmotionDataset(X_val, y_val, self.tokenizer)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=False,
            report_to="none",
            disable_tqdm=False
        )

        def compute_metrics(pred):
            logits, labels = pred
            predictions = torch.argmax(torch.tensor(logits), dim=-1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1_macro': f1_score(labels, predictions, average='macro'),
                'f1_weighted': f1_score(labels, predictions, average='weighted')
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

    def predict(self, texts):
        """Предсказание на новых текстах"""
        dataset = EmotionDataset(texts, [0] * len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16)
        all_preds = []
        all_logits = []

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        return np.array(all_preds), np.array(all_logits)

    def save(self, path="models/rubert_cased_finetuned"):
        """Сохранение дообученной модели"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Модель сохранена в {path}")

    @classmethod
    def load(cls, path="models/rubert_cased_finetuned"):
        """Загрузка модели из файла"""
        print(f"Загрузка модели из {path}")
        return cls(model_path=path)
