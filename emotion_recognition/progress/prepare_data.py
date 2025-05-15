from matplotlib import pyplot as plt

from emotion_recognition.data_load.data_load import read_disk
from emotion_recognition.preprocess.ru_emotions import map_emotions_to_ekman
from emotion_recognition.utils import save_pandas, read_pandas, EKMAN_EMOTIONS


def load_and_save_clean():
    from emotion_recognition.data_load.data_load import read_disk, load_ru_go_emotions
    from emotion_recognition.preprocess.ru_emotions import map_emotions_to_ekman

    load_ru_go_emotions("data/ru_go_emotions_raw")
    df = read_disk("data/ru_go_emotions_raw")["train"].to_pandas()
    df_clean = map_emotions_to_ekman(df)
    save_pandas(df_clean, 'data/ru_go_emotions_ekman.csv')


def preprocess():
    from emotion_recognition.preprocess.ru_emotions import preprocess_text

    df_clean = read_pandas('data/ru_go_emotions_ekman.csv')
    df_clean["tokens"] = df_clean["text"].apply(preprocess_text)
    save_pandas(df_clean, "data/ru_go_emotions_tokens.csv")


def clean_labels():
    from emotion_recognition.preprocess.ru_emotions import single_label
    df_clean = read_pandas("data/ru_go_emotions_tokens.csv", "tokens")
    df_clean[EKMAN_EMOTIONS] = df_clean[EKMAN_EMOTIONS].apply(single_label, axis=1)
    save_pandas(df_clean, "data/ru_go_emotions_tokens_single.csv")


def bars_count():
    df = read_disk("../data/ru_go_emotions_raw")["train"].to_pandas()
    df_mapped = map_emotions_to_ekman(df)[:50000]

    emotion_counts = df_mapped[EKMAN_EMOTIONS].sum().sort_values(ascending=False)

    # Визуализируем
    plt.figure(figsize=(10, 6))
    bars = emotion_counts.plot(kind='bar', color='skyblue')
    plt.title("Количество меток по эмоциям (по Экману)")
    plt.xlabel("Эмоция")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над столбцами
    for bar in bars.patches:
        bars.annotate(f'{int(bar.get_height())}',
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

