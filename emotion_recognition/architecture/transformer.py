import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=10, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, input_dim)  # Для совместимости
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Mean pooling по последовательности
        x = self.dropout(x)
        return self.classifier(x)

def load_transformer_model(model_path, input_dim, num_classes, device='cuda'):
    # Создаем экземпляр модели с теми же параметрами, что были при обучении
    model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes, nhead=10, num_layers=2).to(device)
    # Загружаем сохраненные веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Переводим в режим оценки
    model.eval()
    return model

def train_transformer_model(X_train, y_train, X_val, y_val, input_dim, num_classes,
                            device='cuda', epochs=20, batch_size=64, learning_rate=1e-3):
    model = TransformerClassifier(input_dim, num_classes, nhead=10, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Преобразуем в тензоры
    X_train_t = torch.tensor(X_train).float().to(device)
    y_train_t = torch.tensor(y_train).long().to(device)
    X_val_t = torch.tensor(X_val).float().to(device)
    y_val_t = torch.tensor(y_val).long().to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    patience = 3
    best_val_acc = 0
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        train_acc = correct_train / total_train
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)

        # Валидация
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()

        val_acc = correct_val / total_val
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1} | Loss: {total_loss:.2f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }