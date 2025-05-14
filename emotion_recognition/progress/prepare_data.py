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
