import gradio as gr
from replacer.ui.tools_ui import AttrDict, ResizeHandleRow

from .project import select, init, genNewProjectPath



def makeVideoProjectUI(comp: AttrDict):
    with ResizeHandleRow():
        with gr.Column():
            gr.Markdown("__*Init*__")
            init_video = gr.Textbox(
                label="Init video",
                placeholder="A video on the same machine where the server is running.",
                elem_id="replacer_init_video")
            with gr.Row():
                project_path = gr.Textbox(
                    label="Path to project",
                    elem_id="replacer_video_project_path")
                gen_path = gr.Button("Generate path")
            init_button = gr.Button("Init")

        with gr.Column():
            gr.Markdown("__*Select*__")
            select_path = gr.Textbox(label="Project path")
            select_button = gr.Button("Select")

    gen_path.click(fn=genNewProjectPath, inputs=[init_video], outputs=[project_path])
    init_button.click(fn=init, inputs=[project_path, init_video], outputs=[comp.selected_project_status, comp.selected_project, select_path])
    select_button.click(fn=select, inputs=[select_path], outputs=[comp.selected_project_status, comp.selected_project])

