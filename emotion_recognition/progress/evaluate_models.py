import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from emotion_recognition.architecture.lstm import train_lstm_model
from emotion_recognition.architecture.svm import build_svm_model
from emotion_recognition.architecture.transformer import train_transformer_model
from emotion_recognition.utils import read_pandas, get_cached_mean_vectors, split_data, EKMAN_EMOTIONS, get_cached_seq_vectors, \
    combine_embeddings


def evaluate_svm_with_vectorizers(data_len=None):
    cache_names = {
        "w2v": "w2v_mean",
        "fasttext": "ft_mean",
        "bert": "bert_mean"
    }

    # Создаем папку для логов
    os.makedirs("results/svm", exist_ok=True)
    results = {}

    # Загрузка меток
    y = read_pandas("data/ru_go_emotions_labels.csv")
    y_single = np.argmax(y, axis=1)

    for name, cache_name in cache_names.items():
        print(f"===========Обучение SVM с векторизатором: {name}===========")
        X = get_cached_mean_vectors(cache_name)

        if data_len is not None:
            X = X[:data_len]
            y_single = y_single[:data_len]
        else:
            y_single = y_single

        X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y_single)

        start_time = time.time()
        result = {}

        # Обучение
        model = build_svm_model()
        model.fit(X_train, y_train)

        # Сохраняем модель
        model_path = f"models/svm/svm_model_{name}.joblib"
        os.makedirs("models/svm", exist_ok=True)
        joblib.dump(model, model_path)

        # Оценка
        y_pred_test = model.predict(X_test)
        y_pred_val = model.predict(X_val)

        acc_test = accuracy_score(y_test, y_pred_test)
        acc_val = accuracy_score(y_val, y_pred_val)

        # F1 макро и взвешенный
        f1_test_macro = f1_score(y_test, y_pred_test, average='macro')
        f1_test_weighted = f1_score(y_test, y_pred_test, average='weighted')

        f1_val_macro = f1_score(y_val, y_pred_val, average='macro')
        f1_val_weighted = f1_score(y_val, y_pred_val, average='weighted')

        # F1 по классам
        f1_test_per_class = f1_score(y_test, y_pred_test, average=None)
        f1_val_per_class = f1_score(y_val, y_pred_val, average=None)

        # Сохраняем по классам
        f1_test_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_test_per_class)}
        f1_val_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_val_per_class)}

        result.update({
            "test_accuracy": float(acc_test),
            "val_accuracy": float(acc_val),

            "test_f1_macro": float(f1_test_macro),
            "test_f1_weighted": float(f1_test_weighted),
            "test_f1_per_class": f1_test_dict,

            "val_f1_macro": float(f1_val_macro),
            "val_f1_weighted": float(f1_val_weighted),
            "val_f1_per_class": f1_val_dict,

            "training_time_seconds": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "model_type": "SVM",
            "vectorizer": name,
            "model_path": model_path
        })

        # Сохраняем результаты
        results[name] = result

        # Сохраняем промежуточный результат в JSON
        with open(f"results/svm/results_{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Accuracy на тесте: {acc_test:.4f}")
        print(f"F1-macro на тесте: {f1_test_macro:.4f}")
        print(f"F1-weighted на тесте: {f1_test_weighted:.4f}")

    # Сохраняем все результаты
    with open("results/svm/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Все результаты сохранены в папке 'results/svm'")
    return results


def evaluate_lstm_with_vectorizers(data_len=None):
    cache_names = {
        "w2v": "w2v_seq20_50k",
        "fasttext": "ft_seq20_50k",
        "bert": "bert_seq20_50k"
    }
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    os.makedirs("results/lstm", exist_ok=True)
    os.makedirs("models/lstm", exist_ok=True)

    # Загрузка целевых меток
    y = read_pandas("data/ru_go_emotions_labels.csv")
    y_single = np.argmax(y, axis=1)

    for name, cache_name in cache_names.items():
        print(f"===========Обучение LSTM с векторизатором: {name}===========")

        # Загрузка закешированных последовательностей
        X = get_cached_seq_vectors(cache_name)
        X = X.astype(np.float32)
        if data_len is not None:
            X = X[:data_len]
            y_single_slice = y_single[:data_len]
        else:
            y_single_slice = y_single

        # Разделение выборки
        X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y_single_slice)

        input_dim = X_train.shape[-1]  # размерность эмбеддинга
        num_classes = len(np.unique(y_single_slice))

        # Обучение
        start_time = time.time()
        model, history = train_lstm_model(X_train, y_train, X_val, y_val, input_dim, num_classes, device=device,
                                          epochs=20, hidden_dim=256, num_layers=2)
        train_time = time.time() - start_time

        # Оценка
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long())
        test_loader = DataLoader(test_dataset, batch_size=64)
        correct = 0
        total = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        acc_test = correct / total

        # F1-score
        f1_test_macro = f1_score(all_true, all_preds, average='macro')
        f1_test_weighted = f1_score(all_true, all_preds, average='weighted')
        f1_test_per_class = f1_score(all_true, all_preds, average=None)

        f1_test_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_test_per_class)}

        # То же самое для валидации
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).long())
        val_loader = DataLoader(val_dataset, batch_size=64)
        all_preds_val = []
        all_true_val = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds_val.extend(predicted.cpu().numpy())
                all_true_val.extend(y_batch.cpu().numpy())

        f1_val_macro = f1_score(all_true_val, all_preds_val, average='macro')
        f1_val_weighted = f1_score(all_true_val, all_preds_val, average='weighted')
        f1_val_per_class = f1_score(all_true_val, all_preds_val, average=None)
        f1_val_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_val_per_class)}

        # Сохранение модели
        model_path = f"models/lstm/lstm_{name}.pt"
        torch.save(model.state_dict(), model_path)

        # Сохранение результата
        result = {
            "test_accuracy": float(acc_test),
            "val_accuracy": float(accuracy_score(all_true_val, all_preds_val)),

            "test_f1_macro": float(f1_test_macro),
            "test_f1_weighted": float(f1_test_weighted),
            "test_f1_per_class": f1_test_dict,

            "val_f1_macro": float(f1_val_macro),
            "val_f1_weighted": float(f1_val_weighted),
            "val_f1_per_class": f1_val_dict,

            "training_time_seconds": train_time,
            "timestamp": datetime.now().isoformat(),
            "model_type": "LSTM",
            "vectorizer": name,
            "model_path": model_path,

            # Добавляем историю обучения
            "train_losses": [float(loss) for loss in history["train_losses"]],
            "train_accuracies": [float(acc) for acc in history["train_accuracies"]],
            "val_accuracies": [float(acc) for acc in history["val_accuracies"]],
        }

        results[name] = result

        # Сохраняем JSON
        with open(f"results/lstm/results_{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"LSTM с {name}: Accuracy на тесте = {acc_test:.4f}")
        print(f"F1-macro на тесте = {f1_test_macro:.4f}")
        print(f"F1-weighted на тесте = {f1_test_weighted:.4f}")

    # Сохранить все результаты
    with open("results/lstm/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Все результаты LSTM сохранены в папке 'results/lstm'")
    return results


def evaluate_transformer_with_vectorizers(data_len=None):
    cache_names = {
        "w2v": "w2v_seq20_50k",
        "fasttext": "ft_seq20_50k",
        "bert": "bert_seq20_50k"
    }
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    os.makedirs("results/transformer", exist_ok=True)
    os.makedirs("models/transformer", exist_ok=True)

    # Загрузка целевых меток
    y = read_pandas("data/ru_go_emotions_labels.csv")
    y_single = np.argmax(y, axis=1)

    for name, cache_name in cache_names.items():
        print(f"===========Обучение Transformer с векторизатором: {name}===========")

        # Загрузка закешированных последовательностей
        X = get_cached_seq_vectors(cache_name)
        X = X.astype(np.float32)
        if data_len is not None:
            X = X[:data_len]
            y_single_slice = y_single[:data_len]
        else:
            y_single_slice = y_single

        # Разделение выборки
        X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y_single_slice)

        input_dim = X_train.shape[-1]  # размерность эмбеддинга
        num_classes = len(np.unique(y_single_slice))

        # Обучение
        start_time = time.time()
        model, history = train_transformer_model(
            X_train, y_train, X_val, y_val,
            input_dim, num_classes, device=device,
            epochs=20, batch_size=64
        )
        train_time = time.time() - start_time

        # Оценка
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long())
        test_loader = DataLoader(test_dataset, batch_size=64)
        correct = 0
        total = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        acc_test = correct / total

        # F1-score
        f1_test_macro = f1_score(all_true, all_preds, average='macro')
        f1_test_weighted = f1_score(all_true, all_preds, average='weighted')
        f1_test_per_class = f1_score(all_true, all_preds, average=None)
        f1_test_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_test_per_class)}

        # То же самое для валидации
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).long())
        val_loader = DataLoader(val_dataset, batch_size=64)
        all_preds_val = []
        all_true_val = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds_val.extend(predicted.cpu().numpy())
                all_true_val.extend(y_batch.cpu().numpy())

        f1_val_macro = f1_score(all_true_val, all_preds_val, average='macro')
        f1_val_weighted = f1_score(all_true_val, all_preds_val, average='weighted')
        f1_val_per_class = f1_score(all_true_val, all_preds_val, average=None)
        f1_val_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_val_per_class)}

        # Сохранение модели
        model_path = f"models/transformer/transformer_{name}.pt"
        torch.save(model.state_dict(), model_path)

        # Сохранение результата
        result = {
            "test_accuracy": float(acc_test),
            "val_accuracy": float(accuracy_score(all_true_val, all_preds_val)),

            "test_f1_macro": float(f1_test_macro),
            "test_f1_weighted": float(f1_test_weighted),
            "test_f1_per_class": f1_test_dict,

            "val_f1_macro": float(f1_val_macro),
            "val_f1_weighted": float(f1_val_weighted),
            "val_f1_per_class": f1_val_dict,

            "training_time_seconds": train_time,
            "timestamp": datetime.now().isoformat(),
            "model_type": "Transformer",
            "vectorizer": name,
            "model_path": model_path,

            # Добавляем историю обучения
            "train_losses": [float(loss) for loss in history["train_losses"]],
            "train_accuracies": [float(acc) for acc in history["train_accuracies"]],
            "val_accuracies": [float(acc) for acc in history["val_accuracies"]],
        }

        results[name] = result

        # Сохраняем JSON
        with open(f"results/transformer/results_{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Transformer с {name}: Accuracy на тесте = {acc_test:.4f}")
        print(f"F1-macro на тесте = {f1_test_macro:.4f}")
        print(f"F1-weighted на тесте = {f1_test_weighted:.4f}")

    # Сохранить все результаты
    with open("results/transformer/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Все результаты Transformer сохранены в папке 'results/transformer'")
    return results

def evaluate_transformer_with_vectorizers_merged(data_len=None):
    mean_cache = "ft_mean_50k"
    cache_names = {
        "w2v": "w2v_seq20_50k",
        "fasttext": "ft_seq20_50k",
        "bert": "bert_seq20_50k"
    }
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    os.makedirs("results/transformer_merged", exist_ok=True)
    os.makedirs("models/transformer_merged", exist_ok=True)

    # Загрузка целевых меток
    y = read_pandas("data/ru_go_emotions_labels.csv")
    y_single = np.argmax(y, axis=1)

    for name, cache_name in cache_names.items():
        print(f"===========Обучение Transformer с векторизатором: {name}===========")

        # Загрузка закешированных последовательностей
        X_seq = get_cached_seq_vectors(cache_name)
        X_mean = get_cached_mean_vectors(mean_cache).values
        X = combine_embeddings(X_seq, X_mean)
        X = X.astype(np.float32)

        if data_len is not None:
            X = X[:data_len]
            y_single_slice = y_single[:data_len]
        else:
            y_single_slice = y_single

        # Разделение выборки
        X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y_single_slice)

        input_dim = X_train.shape[-1]  # размерность эмбеддинга
        num_classes = len(np.unique(y_single_slice))

        # Обучение
        start_time = time.time()
        model, history = train_transformer_model(
            X_train, y_train, X_val, y_val,
            input_dim, num_classes, device=device,
            epochs=20, batch_size=64
        )
        train_time = time.time() - start_time

        # Оценка
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).long())
        test_loader = DataLoader(test_dataset, batch_size=64)
        correct = 0
        total = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        acc_test = correct / total

        # F1-score
        f1_test_macro = f1_score(all_true, all_preds, average='macro')
        f1_test_weighted = f1_score(all_true, all_preds, average='weighted')
        f1_test_per_class = f1_score(all_true, all_preds, average=None)
        f1_test_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_test_per_class)}

        # То же самое для валидации
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).long())
        val_loader = DataLoader(val_dataset, batch_size=64)
        all_preds_val = []
        all_true_val = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds_val.extend(predicted.cpu().numpy())
                all_true_val.extend(y_batch.cpu().numpy())

        f1_val_macro = f1_score(all_true_val, all_preds_val, average='macro')
        f1_val_weighted = f1_score(all_true_val, all_preds_val, average='weighted')
        f1_val_per_class = f1_score(all_true_val, all_preds_val, average=None)
        f1_val_dict = {EKMAN_EMOTIONS[i]: float(score) for i, score in enumerate(f1_val_per_class)}

        # Сохранение модели
        model_path = f"models/transformer_merged/transformer_{name}_plus_mean.pt"
        torch.save(model.state_dict(), model_path)

        # Сохранение результата
        result = {
            "test_accuracy": float(acc_test),
            "val_accuracy": float(accuracy_score(all_true_val, all_preds_val)),

            "test_f1_macro": float(f1_test_macro),
            "test_f1_weighted": float(f1_test_weighted),
            "test_f1_per_class": f1_test_dict,

            "val_f1_macro": float(f1_val_macro),
            "val_f1_weighted": float(f1_val_weighted),
            "val_f1_per_class": f1_val_dict,

            "training_time_seconds": train_time,
            "timestamp": datetime.now().isoformat(),
            "model_type": "Transformer",
            "vectorizer": name,
            "model_path": model_path,

            # Добавляем историю обучения
            "train_losses": [float(loss) for loss in history["train_losses"]],
            "train_accuracies": [float(acc) for acc in history["train_accuracies"]],
            "val_accuracies": [float(acc) for acc in history["val_accuracies"]],
        }

        results[name] = result

        # Сохраняем JSON
        with open(f"results/transformer_merged/results_{name}_plus_mean.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Transformer с {name}: Accuracy на тесте = {acc_test:.4f}")
        print(f"F1-macro на тесте = {f1_test_macro:.4f}")
        print(f"F1-weighted на тесте = {f1_test_weighted:.4f}")

    # Сохранить все результаты
    with open("results/transformer_merged/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Все результаты Transformer сохранены в папке 'results/transformer_merged'")
    return results