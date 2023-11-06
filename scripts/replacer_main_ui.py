from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import uuid
import modules.scripts as scripts
import modules.shared
from modules import script_callbacks
import os
import random
import time
import numpy as np
import gradio as gr
from modules.shared import opts, state
from PIL import Image, PngImagePlugin
import torch
from modules import scripts, shared, ui_common, postprocessing, call_queue
from scripts.replacer_generate import generate
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules.ui_common import create_output_panel
from scripts.replacer_options import EXT_NAME, getDetectionPromptExamples, getPositivePromptExamples
from scripts.replacer_options import getNegativePromptExamples, useFirstPositivePromptFromExamples
from scripts.replacer_options import useFirstNegativePromptFromExamples



class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return EXT_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()







def on_ui_tabs():
    with gr.Blocks() as replacer:
        
        tab_index = gr.State(value=0)

        with gr.Row():

            with gr.Column():

                with gr.Row():
                    placeholder = getDetectionPromptExamples()[0]
                    detectionPrompt = gr.Textbox(label="Detection prompt", 
                                        show_label=True, 
                                        lines=1, 
                                        elem_classes=["detectionPrompt"],
                                        placeholder=placeholder)
                    
                    gr.Examples(
                        examples=getDetectionPromptExamples(),
                        inputs=detectionPrompt,
                        label="",
                    )

                with gr.Row():
                    placeholder = None
                    if (useFirstPositivePromptFromExamples()):
                        placeholder = getPositivePromptExamples()[0]
                    
                    positvePrompt = gr.Textbox(label="Positve prompt", 
                                        show_label=True, 
                                        lines=1, 
                                        elem_classes=["positvePrompt"],
                                        placeholder=placeholder)

                    gr.Examples(
                        examples=getPositivePromptExamples(),
                        inputs=positvePrompt,
                        label="",
                        
                    )

                with gr.Row():
                    placeholder = None
                    if (useFirstNegativePromptFromExamples()):
                        placeholder = getNegativePromptExamples()[0]

                    negativePrompt = gr.Textbox(label="Negative prompt", 
                                        show_label=True, 
                                        lines=1, 
                                        elem_classes=["negativePrompt"],
                                        placeholder=placeholder)

                    
                    gr.Examples(
                        examples=getNegativePromptExamples(),
                        inputs=negativePrompt,
                        label="",
                        
                    )

                run_button = gr.Button("Run")      

        
                with gr.Tabs(elem_id="mode_extras"):
                    with gr.TabItem('Single Image', id="single_image", elem_id="single_tab") as tab_single:
                        image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="image", image_mode="RGBA")

                    with gr.TabItem('Batch Process', id="batch_process", elem_id="batch_process_tab") as tab_batch:
                        image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                    with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="batch_directory_tab") as tab_batch_dir:
                        input_batch_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="input_batch_dir")
                        output_batch_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="output_batch_dir")
                        show_batch_dir_results = gr.Checkbox(label='Show result images', value=True, elem_id="show_batch_dir_results")



                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        pass


            with gr.Column():
                img2img_gallery, generation_info, html_info, html_log = create_output_panel("img2img", opts.outdir_img2img_samples)

        tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
        tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
        tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])

        run_button.click(
            fn=wrap_gradio_gpu_call(generate, extra_outputs=[None, '', '']),
            inputs=[
                detectionPrompt,
                positvePrompt,
                negativePrompt,
                tab_index,
                image,
                image_batch,
                input_batch_dir,
                output_batch_dir,
                show_batch_dir_results,
            ],
            outputs=[
                img2img_gallery,
                generation_info,
                html_info,
                html_log,
            ]
        )


    return [(replacer, EXT_NAME, EXT_NAME)]



script_callbacks.on_ui_tabs(on_ui_tabs)


