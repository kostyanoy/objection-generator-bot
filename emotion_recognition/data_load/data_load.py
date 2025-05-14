from datasets import load_dataset, load_from_disk


def load_ru_go_emotions(target_path: str = None):
    """Скачивает датасет и сохраняет его в указанную директорию"""
    dataset = load_dataset("seara/ru_go_emotions", "raw")
    dataset.save_to_disk(target_path)
    print(f"Датасет сохранён в {target_path}")
    return dataset


def read_disk(source_path: str):
    """Загружает датасет с диска"""
    dataset = load_from_disk(source_path)
    print("Датасет успешно загружен с диска")
    return dataset


if __name__ == "__main__":
    load_ru_go_emotions("../data/ru_go_emotions_raw")
