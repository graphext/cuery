import gradio as gr
from .chat import ai_chat, chat


def make_ui():
    with gr.Blocks(theme=gr.themes.Citrus()) as ui:
        # Predefine but render later in different order
        schema_preview = gr.Code("...", language="json", container=True, render=False)
        example_preview = gr.DataFrame(render=False)
        # example_preview = gr.Markdown("...", container=True, render=False)

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("# Chat", container=False)
            with gr.Column():
                gr.Markdown("# Schema Preview", container=False)
        with gr.Row(equal_height=True):
            with gr.Column():
                chatui = gr.ChatInterface(
                    fn=ai_chat,
                    type="messages",
                    additional_outputs=[schema_preview, example_preview],
                    examples=[
                        ["I need a schema for extracting product information"],
                        ["Help me structure user profile data"],
                        ["I want to extract research paper metadata"],
                    ],
                )
            with gr.Column():
                schema_preview.render()
        with gr.Row():
            gr.Markdown("# Table Preview", container=False)
        with gr.Row():
            example_preview.render()

    return ui


def launch():
    ui = make_ui()
    ui.launch()
