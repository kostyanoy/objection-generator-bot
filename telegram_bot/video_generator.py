import asyncio
import json
import logging
import os
from asyncio import AbstractEventLoop
from datetime import datetime
from threading import Timer

from telegram import Bot, Update, Message

from telegram_bot.emotion_classifier import EmotionClassifier
from video_synthesis.Phrase import Phrase
from video_synthesis.main import render_dialog


class VideoGenerator:
    MAX_MESSAGES = 50

    def __init__(self, loop: AbstractEventLoop, bot: Bot, chat_id, classifier: EmotionClassifier):
        self.loop = loop
        self.bot: Bot = bot
        self.chat_id = chat_id
        self.messages: list[Message] = []
        self.is_processing = False
        self.timer: Timer | None = None
        self.classifier = classifier
        self.config_path_base = "configs"
        self.results_path_base = "results"
        self.logger = logging.getLogger()

        os.makedirs(self.config_path_base, exist_ok=True)
        os.makedirs(self.results_path_base, exist_ok=True)

    def add_message(self, update: Update):
        if self.is_processing:
            self.logger.info(f"Отклонено сообщение для {self.chat_id}, уже обрабатываются")
            return
        # дожидаемся сбора всех пересланных сообщений
        if self.timer is not None:
            if len(self.messages) >= self.MAX_MESSAGES:
                self.logger.info(f"Достигнут лимит по количеству пересланных сообщений для {self.chat_id}")
                self._send_message(f"Слишком много сообщений, максимум {self.MAX_MESSAGES}...")
                return
            else:
                self.logger.info(f"Обновлен таймер для {self.chat_id}")
                self.timer.cancel()
        self.messages.append(update.message)
        self.timer = Timer(3.0, self._generate_video)
        self.timer.start()
        self.logger.info(f"Добавлено сообщение для {self.chat_id}: {update.message}")

    def _generate_video(self):
        path = None
        try:
            self.logger.info(f"Начата обработка видео для {self.chat_id}")

            self.is_processing = True
            self.timer = None

            self._send_message("Сообщения приняты! Начинается обработка...")

            # собираем все сообщения вместе
            scene = self._get_scene()
            phrases = [Phrase(id=p["id"], name=p["sender_name"], text=p["text"],
                              emotion=self.classifier.get_score(p["emotion"])) for p in scene]

            # рендерим видео
            _id = self._generate_timestamp()
            path = f"{self.results_path_base}/out_{_id}.mp4"
            try:
                pass
                render_dialog(dialog=phrases, output=path)
            except Exception as e:
                self.logger.error(f"Ошибка при генерации видео: {e}", exc_info=True)
                self._send_message("Ошибка при создании видео.")
                return
            self._send_video(video_path=path)

            self.logger.info(f"Закончена обработка видео для {self.chat_id}")
        except Exception as e:
            self.logger.error(f"Критическая ошибка: {e}", exc_info=True)
            self._send_message("Произошла непредвиденная ошибка.")
        finally:
            # очищаем дисковое пространство
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    self.logger.info(f"Видео файл удален: {path}")
                except Exception as e:
                    self.logger.warning(f"Не удалось удалить видео файл: {e}")
            self.messages = []
            self.is_processing = False

    def _get_scene(self):
        # Собираем диалог
        dialog = []
        for msg in self.messages:
            if "forward_from" in msg.api_kwargs:
                sender_user: dict = msg.api_kwargs["forward_from"]
                sender_id = sender_user["id"]
                sender_name = sender_user.get("first_name", sender_user.get("username", "hidden"))
            elif "forward_sender_name" in msg.api_kwargs:
                sender_id = msg.api_kwargs["forward_sender_name"]
                sender_name = msg.api_kwargs["forward_sender_name"]
            else:
                sender_id = 0
                sender_name = "hidden"

            emotion = self.classifier.classify(msg.text)
            dialog.append({
                "id": sender_id,
                "text": msg.text,
                "emotion": emotion,
                "sender_name": sender_name + f" ({emotion})",
            })

        # Формируем JSON для генератора видео
        config = {
            "dialog": dialog
        }
        task_id = self._generate_timestamp()
        config_path = f"{self.config_path_base}/task_{task_id}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return dialog

    def _generate_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _send_message(self, text):
        asyncio.run_coroutine_threadsafe(
            self.bot.send_message(chat_id=self.chat_id, text=text),
            self.loop
        )

    def _send_video(self, video_path):
        if not os.path.exists(video_path):
            self.logger.error(f"Видео файл не найден: {video_path}")
            self._send_message("Ошибка: видеофайл не найден.")
        with open(video_path, 'rb') as video_file:
            asyncio.run_coroutine_threadsafe(
                self.bot.send_video(chat_id=self.chat_id, video=video_file),
                self.loop
            ).result()  # result() нужен для синхронного ожидания выполнения
        self.logger.info(f"Видео успешно отправлено для чата {self.chat_id}")
