from objection_engine import render_comment_list
from objection_engine.beans.comment import Comment

from video_synthesis.Phrase import Phrase


def render_dialog(dialog: list[Phrase], output: str):
    comments = [
        Comment(user_id=p.id, user_name=p.name, text_content=p.text, score=p.emotion)
        for p in dialog
    ]

    render_comment_list(
        comment_list=comments,
        output_filename=output,
        resolution_scale=2
    )

