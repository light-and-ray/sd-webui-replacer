import gradio as gr
from modules import scripts, shared, sd_samplers, ui_toprow, ui, script_callbacks
from modules.shared import cmd_opts
from modules.ui_components import ToolButton, ResizeHandleRow
from modules.call_queue import wrap_gradio_gpu_call
from modules.ui_common import create_output_panel, refresh_symbol, update_generation_info
from replacer.generate import generate_webui, applyHiresFix_webui, getLastUsedSeed
from replacer.options import (EXT_NAME, EXT_NAME_LOWER, getSaveDir, getDetectionPromptExamples,
    getPositivePromptExamples, getNegativePromptExamples, useFirstPositivePromptFromExamples,
    useFirstNegativePromptFromExamples, getHiresFixPositivePromptSuffixExamples,
    needHideSegmentAnythingAccordions, needAutoUnloadModels, getAvoidancePromptExamples,
)
from replacer import replacer_scripts


try:
    from modules.ui_common import OutputPanel # webui 1.8+
    OUTPUT_PANEL_AVALIABLE = True
except Exception as e:
    OUTPUT_PANEL_AVALIABLE = False



def hideSegmantAnythingAccordions(demo, app):
    try:
        for tab in ['txt2img', 'img2img']:
            samUseCpuPath = f"{tab}/Use CPU for SAM/value"
            samUseCpu = demo.ui_loadsave.component_mapping[samUseCpuPath]
            accordion = samUseCpu.parent.parent.parent.parent
            accordion.visible = False
            accordion.render = False
        print(f"[{EXT_NAME}] Segment Anythings accordions are hidden")
    except Exception as e:
        print(f"[{EXT_NAME}] not possible to hide Segment Anythings accordions: {e}")


def getSubmitJsFunction(galleryId, buttonsId):
    return 'function(){'\
        'var arguments_ = Array.from(arguments);'\
        f'arguments_.push("{buttonsId}", "{galleryId}");'\
        'return submit_replacer.apply(null, arguments_);'\
    '}'


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

        tab_index = gr.Number(value=0, visible=False)
        dummy_component = gr.Label(visible=False)
        replacer_scripts.initCNScript()

        with ResizeHandleRow():

            with gr.Column():

                with gr.Row():
                    placeholder = getDetectionPromptExamples()[0]
                    detectionPrompt = gr.Textbox(label="Detection prompt",
                                        show_label=True,
                                        lines=1,
                                        elem_classes=["detectionPrompt"],
                                        placeholder=placeholder,
                                        elem_id="replacer_detectionPrompt")

                    gr.Examples(
                        examples=getDetectionPromptExamples(),
                        inputs=detectionPrompt,
                        label="",
                        elem_id="replacer_detectionPrompt_examples",
                    )

                with gr.Row():
                    avoidancePrompt = gr.Textbox(label="Avoidance prompt",
                                        show_label=True,
                                        lines=1,
                                        elem_classes=["avoidancePrompt"],
                                        placeholder=None,
                                        elem_id="replacer_avoidancePrompt")

                    gr.Examples(
                        examples=getAvoidancePromptExamples(),
                        inputs=avoidancePrompt,
                        label="",
                        elem_id="replacer_avoidancePrompt_examples",
                    )

                with gr.Row():
                    placeholder = None
                    if (useFirstPositivePromptFromExamples()):
                        placeholder = getPositivePromptExamples()[0]

                    positvePrompt = gr.Textbox(label="Positve prompt",
                                        show_label=True,
                                        lines=1,
                                        elem_classes=["positvePrompt"],
                                        placeholder=placeholder,
                                        elem_id="replacer_positvePrompt")

                    gr.Examples(
                        examples=getPositivePromptExamples(),
                        inputs=positvePrompt,
                        label="",
                        elem_id="replacer_positvePrompt_examples",
                    )

                with gr.Row():
                    placeholder = None
                    if (useFirstNegativePromptFromExamples()):
                        placeholder = getNegativePromptExamples()[0]

                    negativePrompt = gr.Textbox(label="Negative prompt",
                                        show_label=True,
                                        lines=1,
                                        elem_classes=["negativePrompt"],
                                        placeholder=placeholder,
                                        elem_id="replacer_negativePrompt")


                    gr.Examples(
                        examples=getNegativePromptExamples(),
                        inputs=negativePrompt,
                        label="",
                        elem_id="replacer_negativePrompt_examples",
                    )

                toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part='replacer')
                toprow.create_inline_toprow_image()
                run_button = toprow.submit
                run_button.variant = 'secondary'
                run_button.value = 'Run'


                with gr.Accordion("Advanced options", open=False):
                    with gr.Row():
                        sampler = gr.Dropdown(
                            label='Sampling method',
                            elem_id="replacer_sampler",
                            choices=sd_samplers.visible_sampler_names(),
                            value="DPM++ 2M SDE Karras"
                        )

                        steps = gr.Slider(
                            label='Steps',
                            value=20,
                            step=1,
                            minimum=1,
                            maximum=150,
                            elem_id="replacer_steps"
                        )

                    with gr.Row():
                        box_threshold = gr.Slider(label='Box Threshold',
                            value=0.3, elem_id="replacer_box_threshold",
                            minimum=0.0, maximum=1.0, step=0.05)
                        mask_expand = gr.Slider(label='Mask Expand',
                            value=35, elem_id="replacer_mask_expand",
                            minimum=-100, maximum=100, step=1)
                        mask_blur = gr.Slider(label='Mask Blur',
                            value=4, elem_id="replacer_mask_blur",
                            minimum=0, maximum=10, step=1)

                    with gr.Row():

                        if not needAutoUnloadModels():
                            unload = gr.Button(
                                value="Unload detection models",
                                elem_id="replacer_unload_detection_models")

                        max_resolution_on_detection = gr.Slider(
                            label='Max resolution on detection',
                            value=1280,
                            step=1,
                            minimum=64,
                            maximum=2560,
                            elem_id="replacer_max_resolution_on_detection"
                        )

                    with gr.Row():
                        from scripts.sam import sam_model_list, refresh_sam_models
                        from scripts.dino import dino_model_list

                        sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list,
                            value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                        sam_refresh_models = ToolButton(value=refresh_symbol)
                        sam_refresh_models.click(refresh_sam_models, sam_model_name,sam_model_name)

                        dino_model_name = gr.Dropdown(label="GroundingDINO Model", choices=dino_model_list, value=dino_model_list[0])

                    with gr.Row():
                        cfg_scale = gr.Slider(label='CFG Scale',
                            value=5.5, elem_id="replacer_cfg_scale",
                            minimum=1.0, maximum=30.0, step=0.5)
                        denoise = gr.Slider(label='Denoising',
                            value=1.0, elem_id="replacer_denoise",
                            minimum=0.0, maximum=1.0, step=0.01)
                        inpaint_padding = gr.Slider(label='Padding',
                            value=40, elem_id="replacer_inpaint_padding",
                            minimum=0, maximum=250, step=1)

                    with gr.Row():
                        inpainting_fill = gr.Radio(label='Masked content',
                            choices=['fill', 'original', 'latent noise', 'latent nothing'],
                            value='fill', type="index", elem_id="replacer_inpainting_fill")

                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(label='width',
                                value=512, elem_id="replacer_width",
                                minimum=64, maximum=2048, step=8)
                            height = gr.Slider(label='height',
                                value=512, elem_id="replacer_height",
                                minimum=64, maximum=2048, step=8)
                        with gr.Column():
                            batch_count = gr.Slider(label='batch count',
                                value=1, elem_id="replacer_batch_count",
                                minimum=1, maximum=10, step=1)
                            batch_size = gr.Slider(label='batch size',
                                value=1, elem_id="replacer_batch_size",
                                minimum=1, maximum=10, step=1)

                    with gr.Row():
                        upscalerForImg2Img = gr.Dropdown(
                            value=None,
                            choices=[x.name for x in shared.sd_upscalers],
                            label="Upscaler for img2Img",
                            elem_id="replacer_upscalerForImg2Img",
                        )

                        if cmd_opts.use_textbox_seed:
                            seed = gr.Textbox(label='Seed', value="", elem_id="replacer_seed", min_width=100)
                        else:
                            seed = gr.Number(label='Seed', value=-1, elem_id="replacer_seed", min_width=100, precision=0)

                        random_seed = ToolButton(
                            ui.random_symbol,
                            elem_id="replacer_random_seed",
                            label='Random seed'
                        )
                        reuse_seed = ToolButton(
                            ui.reuse_symbol,
                            elem_id="replacer_reuse_seed",
                            label='Reuse seed'
                        )

                    with gr.Row():
                        inpainting_mask_invert = gr.Radio(
                            label='Mask mode',
                            choices=['Inpaint masked', 'Inpaint not masked'],
                            value='Inpaint masked',
                            type="index",
                            elem_id="replacer_mask_mode")

                    with gr.Row():
                        save_grid = gr.Checkbox(label='Save grid for batch size/count',
                            value=False, elem_id="replacer_save_grid")
                        extra_includes = gr.CheckboxGroup(
                            choices=["mask", "box", "cutted", "preview", "script"],
                            label="Extra include in gallery",
                            type="value",
                            elem_id="replacer_extra_includes",
                            value=[],
                        )


                with gr.Tabs(elem_id="replacer_input_modes"):
                    with gr.TabItem('Single Image', id="single_image", elem_id="replacer_single_tab") as tab_single:
                        image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="image", image_mode="RGBA")

                    with gr.TabItem('Batch Process', id="batch_process", elem_id="replacer_batch_process_tab") as tab_batch:
                        image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="replacer_image_batch")

                    with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="replacer_batch_directory_tab") as tab_batch_dir:
                        input_batch_dir = gr.Textbox(
                            label="Input directory", **shared.hide_dirs,
                            placeholder="A directory on the same machine where the server is running.",
                            elem_id="replacer_input_batch_dir")
                        output_batch_dir = gr.Textbox(
                            label="Output directory", **shared.hide_dirs,
                            placeholder="Leave blank to save images to the default path.",
                            elem_id="replacer_output_batch_dir")
                        show_batch_dir_results = gr.Checkbox(
                            label='Show result images', value=False, elem_id="replacer_show_batch_dir_results")

                    with gr.TabItem('Video', id="batch_from_video", elem_id="replacer_batch_video_tab") as tab_batch_video:
                        input_video = gr.Textbox(
                            label="Input video",
                            placeholder="A video on the same machine where the server is running.",
                            elem_id="replacer_input_video")
                        target_video_fps = gr.Slider(
                            label='FPS', value=10.0, min=0.0, step=0.1, max=60.0,
                            info="(0 = fps from input video)",
                            elem_id="replacer_video_fps")
                        output_batch_dir = gr.Textbox(
                            label="Output directory", **shared.hide_dirs,
                            placeholder="Leave blank to save images to the default path.",
                            info='(default is the same directory with input video. Rusult is in "output_seed" subdirectory)',
                            elem_id="replacer_output_batch_dir")
                        gr.Markdown("To increase consistency it's better to inpaint clear "\
                            "objects on video with good quality and enough context. "\
                            "Your prompts need to produce consistent results\n\n"\
                            "To suppress flickering you can generate in little fps (e.g. 10), "\
                            "then interpolate (x2) it with ai interpolation algorithm "\
                            "(e.g [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) or "\
                            "[frame interpolation in deforum sd-webui extension]("\
                            "https://github.com/deforum-art/sd-webui-deforum/wiki/Upscaling-and-Frame-Interpolation))")
                
                cn_inputs = []
                if replacer_scripts.script_controlnet:
                    cn_inputs = list(replacer_scripts.script_controlnet.ui(True))

            with gr.Column():
                with gr.Row():
                    if OUTPUT_PANEL_AVALIABLE:
                        outputPanel = create_output_panel('replacer', getSaveDir())
                        img2img_gallery = outputPanel.gallery
                        generation_info = outputPanel.generation_info
                        html_info = outputPanel.infotext
                        html_log = outputPanel.html_log
                    else:
                        img2img_gallery, generation_info, html_info, html_log = \
                            create_output_panel('replacer', getSaveDir())
                    generation_info_button = gr.Button(visible=False, elem_id="replacer_generation_info_button")
                    generation_info_button.click(
                        fn=update_generation_info,
                        _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                        inputs=[generation_info, html_info, html_info],
                        outputs=[html_info, html_info],
                        show_progress=False,
                    )

                with gr.Row():
                    toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part=f'replacer_hf')
                    toprow.create_inline_toprow_image()
                    apply_hires_fix_button = toprow.submit
                    apply_hires_fix_button.variant = 'secondary'
                    apply_hires_fix_button.value = 'Apply HiresFix'

                with gr.Row():
                    with gr.Accordion("HiresFix options", open=False):
                        with gr.Row():
                            hf_upscaler = gr.Dropdown(
                                value="ESRGAN_4x",
                                choices=[x.name for x in shared.sd_upscalers],
                                label="Upscaler",
                            )

                            hf_steps = gr.Slider(
                                label='Hires steps',
                                value=4,
                                step=1,
                                minimum=0,
                                maximum=150,
                                elem_id="replacer_hf_steps"
                            )

                            hf_cfg_scale = gr.Slider(
                                label='Hires CFG Scale',
                                value=1.0,
                                step=0.5,
                                minimum=1.0,
                                maximum=30.0,
                                elem_id="replacer_hf_cfg_scale"
                            )

                        with gr.Row():
                            hf_sampler = gr.Dropdown(
                                label='Hires sampling method',
                                elem_id="replacer_hf_sampler",
                                choices=["Use same sampler"] + sd_samplers.visible_sampler_names(),
                                value="Use same sampler"
                            )

                            hf_denoise = gr.Slider(
                                label='Hires Denoising',
                                value=0.35,
                                step=0.01,
                                minimum=0.0,
                                maximum=1.0,
                                elem_id="replacer_hf_denoise",
                            )

                        with gr.Row():
                            placeholder = None
                            placeholder = getHiresFixPositivePromptSuffixExamples()[0]

                            hfPositivePromptSuffix = gr.Textbox(
                                label="Suffix for positive prompt",
                                show_label=True,
                                lines=1,
                                elem_classes=["hfPositivePromptSuffix"],
                                placeholder=placeholder,
                                elem_id="replacer_hfPositivePromptSuffix",
                            )

                            gr.Examples(
                                examples=getHiresFixPositivePromptSuffixExamples(),
                                inputs=hfPositivePromptSuffix,
                                label="",
                                elem_id="replacer_hfPositivePromptSuffix_examples",
                            )

                        with gr.Row():
                            hf_size_limit = gr.Slider(
                                label='Limit render size',
                                value=2000,
                                step=1,
                                minimum=1000,
                                maximum=10000,
                                elem_id="replacer_hf_size_limit",
                            )

                            hf_above_limit_upscaler = gr.Dropdown(
                                value="Lanczos",
                                choices=[x.name for x in shared.sd_upscalers],
                                label="Above limit upscaler",
                            )

                            hf_unload_detection_models = gr.Checkbox(
                                label='Unload detection models before hires fix',
                                value=True,
                                elem_id="replacer_hf_unload_detection_models",
                            )
                            if needAutoUnloadModels():
                                hf_unload_detection_models.visible = False

                        with gr.Row():
                            hf_disable_cn = gr.Checkbox(
                                label='Disable ControlNet while hires. fix',
                                value=True,
                                elem_id="replacer_hf_disable_cn",
                            )
                            if not replacer_scripts.script_controlnet:
                                hf_disable_cn.visible = False



        def tab_single_on_select():
            return 0, gr.Button.update(visible=True)

        def tab_batch_on_select():
            return 1, gr.Button.update(visible=False)

        def tab_batch_dir_on_select():
            return 2, gr.Button.update(visible=False)

        def tab_batch_video_on_select():
            return 3, gr.Button.update(visible=False)

        tab_single.select(fn=tab_single_on_select, inputs=[], outputs=[tab_index, apply_hires_fix_button])
        tab_batch.select(fn=tab_batch_on_select, inputs=[], outputs=[tab_index, apply_hires_fix_button])
        tab_batch_dir.select(fn=tab_batch_dir_on_select, inputs=[], outputs=[tab_index, apply_hires_fix_button])
        tab_batch_video.select(fn=tab_batch_video_on_select, inputs=[], outputs=[tab_index, apply_hires_fix_button])


        run_button.click(
            _js=getSubmitJsFunction('replacer', 'replacer'),
            fn=wrap_gradio_gpu_call(generate_webui, extra_outputs=[None, '', '']),
            inputs=[
                dummy_component,
                detectionPrompt,
                avoidancePrompt,
                positvePrompt,
                negativePrompt,
                tab_index,
                image,
                image_batch,
                input_batch_dir,
                output_batch_dir,
                show_batch_dir_results,
                input_video,
                target_video_fps,
                upscalerForImg2Img,
                seed,
                sampler,
                steps,
                box_threshold,
                mask_expand,
                mask_blur,
                max_resolution_on_detection,
                sam_model_name,
                dino_model_name,
                cfg_scale,
                denoise,
                inpaint_padding,
                inpainting_fill,
                width,
                batch_count,
                height,
                batch_size,
                inpainting_mask_invert,
                save_grid,
                extra_includes,
            ] + cn_inputs,
            outputs=[
                img2img_gallery,
                generation_info,
                html_info,
                html_log,
            ],
            show_progress=False,
        )


        apply_hires_fix_button.click(
            _js=getSubmitJsFunction('replacer', f'replacer_hf'),
            fn=wrap_gradio_gpu_call(applyHiresFix_webui, extra_outputs=[None, '', '']),
            inputs=[
                dummy_component,
                hf_upscaler,
                hf_steps,
                hf_sampler,
                hf_denoise,
                hf_cfg_scale,
                hfPositivePromptSuffix,
                hf_size_limit,
                hf_above_limit_upscaler,
                hf_unload_detection_models,
                hf_disable_cn,
            ],
            outputs=[
                img2img_gallery,
                generation_info,
                html_info,
                html_log,
            ],
            show_progress=False,
        )


        random_seed.click(
            fn=lambda: -1,
            inputs=[
            ],
            outputs=[
                seed,
            ]
        )

        reuse_seed.click(
            fn=getLastUsedSeed,
            inputs=[
            ],
            outputs=[
                seed,
            ]
        )

        if not needAutoUnloadModels():
            from scripts.sam import clear_cache
            unload.click(
                fn=clear_cache,
                inputs=[],
                outputs=[])


    return [(replacer, EXT_NAME, EXT_NAME)]



script_callbacks.on_ui_tabs(on_ui_tabs)

if needHideSegmentAnythingAccordions():
    script_callbacks.on_app_started(hideSegmantAnythingAccordions)


