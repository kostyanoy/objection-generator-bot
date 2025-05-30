from objection_engine.renderer import render_comment_list
from objection_engine.beans.comment import Comment

comments = [
    Comment(
        user_name="Phoenix",
        text_content="Hello. My name is Phoenix. I am a defense attorney.",
    ),
    Comment(user_name="Phoenix", text_content="Here is another line of dialogue."),
    Comment(
        user_name="Edgeworth",
        text_content="I am Edgeworth, because I have the second-most lines.",
    ),
]

render_comment_list(comments, output_filename="out.mp4")