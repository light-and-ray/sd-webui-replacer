import gradio as gr
from modules import shared, ui_settings
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import AttrDict, ResizeHandleRow



def makeVideoOptionsUI(comp: AttrDict):
    gr.Markdown("**This options are not saved in the project**")
    comp.target_video_fps = gr.Slider(
        label='FPS', value=15.0, step=0.1, minimum=0.0, maximum=100.0,
        elem_id="replacer_video_fps")

    comp.selected_video_mode = gr.Textbox(value="video_mode_animatediff", visible=False)


    with gr.Tabs(elem_id="replacer_video_modes"):

        with gr.Tab("AnimateDiff mode", elem_id="replacer_video_mode_animatediff") as comp.video_mode_animatediff:
            with ResizeHandleRow():
                with gr.Column():

                    if replacer_extensions.controlnet.SCRIPT is None:
                        gr.Markdown("[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) is not installed")
                        if replacer_extensions.animatediff.SCRIPT is None:
                            gr.Markdown("[sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) is not installed")

                    with gr.Row():
                        comp.ad_fragment_length = gr.Number(
                            minimum=0, value=36, precision=0,
                            label="Fragment length, frames", info="Set 0 to consider the full video as 1 fragment",
                            elem_id="replacer_ad_fragment_length",
                        )
                        comp.ad_internal_fps = gr.Number(
                            value=0, precision=0, info="Set 0 to use the same with input video",
                            minimum=0,
                            label="Internal AD FPS",
                            elem_id="replacer_ad_internal_fps"
                        )

                    with gr.Row():
                        comp.ad_batch_size = gr.Slider(
                            minimum=1, maximum=32, value=12,
                            label="Context batch size",
                            step=1, precision=0,
                            elem_id="replacer_ad_batch_size",
                        )
                        comp.ad_stride = gr.Number(
                            minimum=1, value=1,
                            label="Stride", precision=0,
                            elem_id="replacer_ad_stride",
                        )
                        comp.ad_overlap = gr.Number(
                            minimum=-1, value=-1,
                            label="Overlap", precision=0,
                            elem_id="replacer_ad_overlap",
                        )

                    with gr.Row():
                        comp.ad_latent_power = gr.Slider(
                            minimum=0.1, maximum=10, value=1.0,
                            step=0.1, label="Latent power",
                            elem_id="replacer_ad_latent_power",
                        )
                        comp.ad_latent_scale = gr.Slider(
                            minimum=1, maximum=128, value=16,
                            label="Latent scale",
                            elem_id="replacer_ad_latent_scale"
                        )

                    with gr.Row():
                        comp.ad_generate_only_first_fragment = gr.Checkbox(
                            label='Generate only the first fragment',
                            info="Useful if you want to test animatediff's options",
                            value=False,
                            elem_id="replacer_ad_generate_only_first_fragment",
                        )

                with gr.Column():
                    with gr.Accordion("Models", open=True):
                        with gr.Row():
                            inpaintModels = replacer_extensions.controlnet.getInpaintModels()
                            comp.ad_cn_inpainting_model = gr.Dropdown(
                                choices=inpaintModels,
                                label="Controlnet inpainting model",
                                value=inpaintModels[0],
                                elem_id="replacer_ad_cn_inpainting_model",
                                info="Occupies the last controlnet unit",
                            )
                            comp.ad_control_weight = gr.Slider(
                                label="Control Weight",
                                value=1.0, minimum=0.0, maximum=2.0, step=0.05,
                                elem_id="replacer_ad_control_weight",
                                elem_classes="controlnet_control_weight_slider",
                            )

                        with gr.Row():
                            comp.ad_force_override_sd_model = gr.Checkbox(label='Force override stable diffusion model',
                                value=True, elem_id=f"replacer_ad_force_override_sd_model",
                                info='Be sure you use NON-inpainting model here')
                            comp.ad_force_sd_model_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')

                        with gr.Row():
                            motionModels = replacer_extensions.animatediff.getModels()
                            default = "mm_sd15_v3.safetensors"
                            if default not in motionModels:
                                default = motionModels[0]
                            comp.ad_motion_model = gr.Dropdown(
                                choices=motionModels,
                                value=default,
                                label="Motion module",
                                type="value",
                                elem_id="replacer_ad_motion_model",
                            )

                    with gr.Accordion("Help", open=False):
                        gr.Markdown(
                            "Almost all advanced options work here. Inpaint padding doesn't, because it's ControlNet inpainting. "
                            "Lama cleaner in masked content enables CN inpaint_only+lama module instead of inpaint_only\n\n"
                            \
                            "Due to high AnimateDiff's consistency in comparison with *\"Frame by frame\"* mode "
                            "you can use high `mask blur` and `mask expand`.\n\n"
                            \
                            "Hires fix doesn't work here, and as I think, it basically can't, because it will "
                            "decrease the consistency. But you can use the `upscaler for img2img` option - these "
                            "upscalers work and consistent enough.\n\n"
                            \
                            "To increase consistency between fragments, you can use ControlNet, especially `SparseCtrl`, or try to use "
                            "`Fragment length` = 0 (or just very big) and set up `Context batch size`, `Stride`, `Overlap`. I recommend make "
                            "`Fragment length` few times more then `Context batch size`\n\n"
                            \
                            "`Context batch size` is set up for 12GB VRAM with one additional "
                            "ControlNet unit. If you get OutOfMemory error, decrease it\n\n"
                            \
                            "Read [here](https://github.com/light-and-ray/sd-webui-replacer/blob/master/docs/video.md#animatediff-options) "
                            "about AnimateDiff options\n\n"
                            \
                            "If you know any other good advice, please send them into github issues, I can place them here"
                        )


        with gr.Tab("Frame by frame mode", elem_id="replacer_video_mode_frame_by_frame") as comp.video_mode_frame_by_frame:

            if replacer_extensions.controlnet.SCRIPT:
                comp.previous_frame_into_controlnet = gr.CheckboxGroup(value=[], label='Pass the previous frame into ControlNet',
                    choices=[f"Unit {x}" for x in range(shared.opts.data.get("control_net_unit_count", 3))], elem_id='replacer_previous_frame_into_controlnet')
            else:
                comp.previous_frame_into_controlnet = gr.CheckboxGroup(value=[], visible=False)

            with gr.Accordion("Help", open=True):
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
                    "[lama-cleaner](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content) with (low denoising) "\
                    "extensions to increase consistency, if it fits to your scenario\n\n"\
                    \
                    "Also a good can be to use `Pass the previous frame into ControlNet` "\
                    "with _IP-Adapter_, _Reference_, _Shuffle_, _T2IA-Color_, _T2IA-Style_"
                    )

    comp.video_mode_animatediff.select(fn=lambda: "video_mode_animatediff", inputs=[], outputs=[comp.selected_video_mode])
    comp.video_mode_frame_by_frame.select(fn=lambda: "video_mode_frame_by_frame", inputs=[], outputs=[comp.selected_video_mode])


