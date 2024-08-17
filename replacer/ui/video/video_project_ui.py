import gradio as gr
from modules import shared, ui_settings
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import AttrDict, ResizeHandleRow



def makeVideoProjectUI(comp: AttrDict):
    comp.input_video = gr.Textbox(
        label="Input video",
        placeholder="A video on the same machine where the server is running.",
        elem_id="replacer_input_video")
    comp.video_output_dir = gr.Textbox(
        label="Output directory",
        placeholder="Leave blank to save images to the default path.",
        info='(default is the same directory with input video. Result is in "out_seed_timestamp" subdirectory)',
        elem_id="replacer_video_output_dir")

