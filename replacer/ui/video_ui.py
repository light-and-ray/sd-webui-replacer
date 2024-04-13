import gradio as gr
from modules import shared
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import AttrDict



def makeVideoUI(comp: AttrDict):
    comp.input_video = gr.Textbox(
        label="Input video",
        placeholder="A video on the same machine where the server is running.",
        elem_id="replacer_input_video")
    comp.target_video_fps = gr.Slider(
        label='FPS', value=16.0, step=0.1, minimum=0.0, maximum=100.0, 
        info="(0 = fps from input video)",
        elem_id="replacer_video_fps")
    comp.video_output_dir = gr.Textbox(
        label="Output directory", **shared.hide_dirs,
        placeholder="Leave blank to save images to the default path.",
        info='(default is the same directory with input video. Rusult is in "out_seed_timestamp" subdirectory)',
        elem_id="replacer_video_output_dir")

    comp.selected_video_mode = gr.Textbox(value="video_mode_animatediff", visible=False)

    with gr.Tabs(elem_id="replacer_video_modes"):
        with gr.Tab("AnimateDiff mode", elem_id="replacer_video_mode_animatediff") as comp.video_mode_animatediff:
            gr.Markdown("Coming soon...")

        with gr.Tab("Frame by frame mode", elem_id="replacer_video_mode_frame_by_frame") as comp.video_mode_frame_by_frame:
            if replacer_extensions.controlnet.SCRIPT:
                comp.previous_frame_into_controlnet = gr.CheckboxGroup(value=[], label='Pass the previous frame into ControlNet',
                    choices=[f"Unit {x}" for x in range(shared.opts.data.get("control_net_unit_count", 3))], elem_id='replacer_previous_frame_into_controlnet')
            else:
                comp.previous_frame_into_controlnet = gr.CheckboxGroup(value=[], visible=False)

            with gr.Accordion("Help", open=False):
                gr.Markdown(
                    "To increase consistency it's better to inpaint clear "\
                    "objects on video with good quality and enough context. "\
                    "Your prompts need to produce consistent results\n\n"\
                    \
                    "To suppress flickering you can generate in little fps (e.g. 10), "\
                    "then interpolate (x2) it with ai interpolation algorithm "\
                    "(e.g [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) or "\
                    "[frame interpolation in deforum sd-webui extension]("\
                    "https://github.com/deforum-art/sd-webui-deforum/wiki/Upscaling-and-Frame-Interpolation))\n\n"\
                    \
                    "You can also use [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) or "\
                    "[lama-cleaner](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content) with (low denosing) "\
                    "extensions to increase consistency, if it fits to your scenario\n\n"\
                    \
                    "Also a good can be to use `Pass the previous frame into ControlNet` "\
                    "with _IP-Adapter_, _Reference_, _Shuffle_, _T2IA-Color_, _T2IA-Style_"
                    )

    comp.video_mode_animatediff.select(fn=lambda: "video_mode_animatediff", inputs=[], outputs=[comp.selected_video_mode])
    comp.video_mode_frame_by_frame.select(fn=lambda: "video_mode_frame_by_frame", inputs=[], outputs=[comp.selected_video_mode])


