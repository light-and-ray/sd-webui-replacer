import gradio as gr
from modules import shared, ui_settings, errors
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.ui_common import create_output_panel, update_generation_info
from replacer.ui.generate_ui import generate_ui, getLastUsedSeed
from replacer.ui.apply_hires_fix import applyHiresFix
from replacer.options import (EXT_NAME, getSaveDir, getDetectionPromptExamples,
    getPositivePromptExamples, getNegativePromptExamples, useFirstPositivePromptFromExamples,
    useFirstNegativePromptFromExamples, doNotShowUnloadButton, getDedicatedPagePath,
    getDetectionPromptExamplesNumber, getPositivePromptExamplesNumber, getNegativePromptExamplesNumber
)
from replacer.extensions import replacer_extensions
from replacer.ui.make_advanced_options import makeAdvancedOptions
from replacer.ui.make_hiresfix_options import makeHiresFixOptions
from replacer.ui.tools_ui import ( update_mask_brush_color, get_current_image, unloadModels, AttrDict,
    getSubmitJsFunction, sendBackToReplacer, IS_WEBUI_1_8, OuputPanelWatcher, ui_toprow,
    setCustomScriptSourceForComponents,
)


try:
    from modules.ui_components import ResizeHandleRow
except:
    ResizeHandleRow = gr.Row



class ReplacerMainUI:
    def __init__(self, isDedicatedPage: bool):
        self.replacerTabUI = None
        self.components = AttrDict()
        self.init_tab(isDedicatedPage)

    def init_tab(self, isDedicatedPage: bool):
        comp = AttrDict()
        with gr.Blocks(analytics_enabled=False) as self.replacerTabUI:
            comp.tab_index = gr.Number(value=0, visible=False)
            comp.dummy_component = gr.Label(visible=False)
            comp.trueComponent = gr.Checkbox(value=True, visible=False)
            comp.falseComponent = gr.Checkbox(value=False, visible=False)
            if replacer_extensions.script_controlnet:
                try:
                    cnUiGroupsLenBefore = len(replacer_extensions.ControlNetUiGroup.all_ui_groups)
                except Exception as e:
                    errors.report(f"Cannot init cnUiGroupsLenBefore: {e}", exc_info=True)
                    replacer_extensions.script_controlnet = None

            with ResizeHandleRow():

                with gr.Column(scale=16):

                    with gr.Row():
                        placeholder = getDetectionPromptExamples()[0]
                        comp.detectionPrompt = gr.Textbox(label="Detection prompt",
                                            show_label=True,
                                            lines=1,
                                            elem_classes=["detectionPrompt"],
                                            placeholder=placeholder,
                                            elem_id="replacer_detectionPrompt")

                        gr.Examples(
                            examples=getDetectionPromptExamples(),
                            inputs=comp.detectionPrompt,
                            label="",
                            elem_id="replacer_detectionPrompt_examples",
                            examples_per_page=getDetectionPromptExamplesNumber(),
                        )

                    with gr.Row():
                        placeholder = None
                        if (useFirstPositivePromptFromExamples()):
                            placeholder = getPositivePromptExamples()[0]

                        comp.positvePrompt = gr.Textbox(label="Positive prompt",
                                            show_label=True,
                                            lines=1,
                                            elem_classes=["positvePrompt"],
                                            placeholder=placeholder,
                                            elem_id="replacer_positvePrompt")

                        gr.Examples(
                            examples=getPositivePromptExamples(),
                            inputs=comp.positvePrompt,
                            label="",
                            elem_id="replacer_positvePrompt_examples",
                            examples_per_page=getPositivePromptExamplesNumber(),
                        )

                    with gr.Row():
                        placeholder = None
                        if (useFirstNegativePromptFromExamples()):
                            placeholder = getNegativePromptExamples()[0]

                        comp.negativePrompt = gr.Textbox(label="Negative prompt",
                                            show_label=True,
                                            lines=1,
                                            elem_classes=["negativePrompt"],
                                            placeholder=placeholder,
                                            elem_id="replacer_negativePrompt")


                        gr.Examples(
                            examples=getNegativePromptExamples(),
                            inputs=comp.negativePrompt,
                            label="",
                            elem_id="replacer_negativePrompt_examples",
                            examples_per_page=getNegativePromptExamplesNumber(),
                        )

                    if ui_toprow:
                        toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part='replacer')
                        toprow.create_inline_toprow_image()
                        comp.run_button = toprow.submit
                        comp.run_button.variant = 'secondary'
                        comp.run_button.value = 'Run'
                    else:
                        comp.run_button = gr.Button('Run', elem_id='replacer_generate')



                    with gr.Row():
                        makeAdvancedOptions(comp, isDedicatedPage)



                    with gr.Tabs(elem_id="replacer_input_modes"):
                        with gr.TabItem('Single Image', id="single_image", elem_id="replacer_single_tab") as comp.tab_single:
                            comp.image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="replacer_image", image_mode="RGBA")

                        with gr.TabItem('Batch Process', id="batch_process", elem_id="replacer_batch_process_tab") as comp.tab_batch:
                            comp.image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="replacer_image_batch")
                            comp.keep_original_filenames = gr.Checkbox(
                                label='Keep original filenames', value=True, elem_id="replacer_keep_original_filenames")

                        with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="replacer_batch_directory_tab") as comp.tab_batch_dir:
                            comp.input_batch_dir = gr.Textbox(
                                label="Input directory", **shared.hide_dirs,
                                placeholder="A directory on the same machine where the server is running.",
                                elem_id="replacer_input_batch_dir")
                            comp.output_batch_dir = gr.Textbox(
                                label="Output directory", **shared.hide_dirs,
                                placeholder="Leave blank to save images to the default path.",
                                elem_id="replacer_output_batch_dir")
                            comp.keep_original_filenames_from_dir = gr.Checkbox(
                                label='Keep original filenames (batch from dir)', value=True, elem_id="replacer_keep_original_filenames_from_dir")
                            comp.show_batch_dir_results = gr.Checkbox(
                                label='Show result images', value=False, elem_id="replacer_show_batch_dir_results")

                        with gr.TabItem('Video', id="batch_from_video", elem_id="replacer_batch_video_tab") as comp.tab_batch_video:
                            comp.input_video = gr.Textbox(
                                label="Input video",
                                placeholder="A video on the same machine where the server is running.",
                                elem_id="replacer_input_video")
                            comp.target_video_fps = gr.Slider(
                                label='FPS', value=10.0, step=0.1, minimum=0.0, maximum=60.0, 
                                info="(0 = fps from input video)",
                                elem_id="replacer_video_fps")
                            comp.video_output_dir = gr.Textbox(
                                label="Output directory", **shared.hide_dirs,
                                placeholder="Leave blank to save images to the default path.",
                                info='(default is the same directory with input video. Rusult is in "output_seed" subdirectory)',
                                elem_id="replacer_video_output_dir")
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
                            if replacer_extensions.script_controlnet:
                                comp.previous_frame_into_controlnet = gr.CheckboxGroup(value=[], label='Pass the previous frame into ControlNet',
                                    choices=[f"Unit {x}" for x in range(shared.opts.data.get("control_net_unit_count", 3))], elem_id='replacer_previous_frame_into_controlnet')
                            else:
                                comp.previous_frame_into_controlnet = gr.CheckboxGroup(value=[], visible=False)
                    
                    comp.cn_inputs = []
                    setCustomScriptSourceForComponents("controlnet")
                    if replacer_extensions.script_controlnet:
                        try:
                            with gr.Row():
                                replacer_extensions.needWatchControlNetUI = True
                                comp.cn_inputs = list(replacer_extensions.script_controlnet.ui(True))
                                replacer_extensions.needWatchControlNetUI = False

                                if not replacer_extensions.controlNetAccordion:
                                    errors.report(f"[{EXT_NAME}] controlnet accordion wasn't found", exc_info=True)
                                else:
                                    with replacer_extensions.controlNetAccordion:
                                        with gr.Row():
                                            gr.Markdown('_If you select Inpaint -> inpaint_only, cn inpaint model will be used instead of sd inpainting_')
                        except Exception as e:
                            errors.report(f"Cannot add controlnet accordion {e}", exc_info=True)
                            replacer_extensions.script_controlnet = None
                    setCustomScriptSourceForComponents(None)


                with gr.Column(scale=15):
                    with gr.Row():
                        if IS_WEBUI_1_8:
                            outputPanel = create_output_panel('replacer', getSaveDir())
                            comp.replacer_gallery = outputPanel.gallery
                            comp.generation_info = outputPanel.generation_info
                            comp.html_info = outputPanel.infotext
                            comp.html_log = outputPanel.html_log
                        else:
                            comp.replacer_gallery, comp.generation_info, comp.html_info, comp.html_log = \
                                create_output_panel('replacer', getSaveDir())
                        comp.generation_info_button = gr.Button(visible=False, elem_id="replacer_generation_info_button")
                        comp.generation_info_button.click(
                            fn=update_generation_info,
                            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
                            inputs=[comp.generation_info, comp.html_info, comp.html_info],
                            outputs=[comp.html_info, comp.html_info],
                            show_progress=False,
                        )
                        if isDedicatedPage and OuputPanelWatcher.send_to_img2img:
                            OuputPanelWatcher.send_to_img2img.visible = False
                            OuputPanelWatcher.send_to_inpaint.visible = False
                            OuputPanelWatcher.send_to_extras.visible = False

                    with gr.Row():
                        if ui_toprow:
                            toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part='replacer_hf')
                            toprow.create_inline_toprow_image()
                            comp.apply_hires_fix_button = toprow.submit
                            comp.apply_hires_fix_button.variant = 'secondary'
                            comp.apply_hires_fix_button.value = 'Apply HiresFix ✨'
                        else:
                            comp.apply_hires_fix_button = gr.Button('Apply HiresFix ✨', elem_id='replacer_hf_generate')



                    with gr.Row():
                        makeHiresFixOptions(comp)



                    with gr.Row():
                        comp.pass_into_hires_fix_automatically = gr.Checkbox(
                                        label='Pass into hires fix automatically',
                                        value=False,
                                        elem_id="replacer_pass_into_hires_fix_automatically",
                                    )
                        comp.save_before_hires_fix = gr.Checkbox(
                                        label='Save images before hires fix',
                                        value=False,
                                        elem_id="replacer_save_before_hires_fix",
                                    )

                    with gr.Row():
                        if not isDedicatedPage:
                            gr.Markdown(f'[Open dedicated page]({getDedicatedPagePath()})')
                        else:
                            comp.sd_model_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')
                            comp.override_sd_model = gr.Checkbox(label='Override sd model dedicated',
                                value=True, elem_id="replacer_override_sd_model", visible=False)



            if replacer_extensions.script_controlnet:
                try:
                    replacer_extensions.ControlNetUiGroup.a1111_context.img2img_w_slider = comp.width
                    replacer_extensions.ControlNetUiGroup.a1111_context.img2img_h_slider = comp.height

                    for ui_group in replacer_extensions.ControlNetUiGroup.all_ui_groups[cnUiGroupsLenBefore:]:
                        ui_group.register_run_annotator()
                        if not replacer_extensions.IS_SD_WEBUI_FORGE:
                            ui_group.inpaint_crop_input_image.value = True
                            ui_group.inpaint_crop_input_image.visible = True
                            ui_group.inpaint_crop_input_image.label = "Crop input image based on generated mask",
                        # if isDedicatedPage: 
                        #     replacer_extensions.ControlNetUiGroup.a1111_context.setting_sd_model_checkpoint = sd_model_checkpoint
                        # ui_group.register_sd_version_changed()
                except Exception as e:
                    errors.report(f"Cannot change ControlNet accordion entry: {e}", exc_info=True)
                    replacer_extensions.script_controlnet = None


            comp.tab_single.select(fn=lambda: 0, inputs=[], outputs=[comp.tab_index])
            comp.tab_batch.select(fn=lambda: 1, inputs=[], outputs=[comp.tab_index])
            comp.tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[comp.tab_index])
            comp.tab_batch_video.select(fn=lambda: 3, inputs=[], outputs=[comp.tab_index])


            comp.run_button.click(
                _js=getSubmitJsFunction('replacer', 'replacer', 'replacer_hf', False),
                fn=wrap_gradio_gpu_call(generate_ui, extra_outputs=[None, '', '']),
                inputs=[
                    comp.dummy_component, # task_id
                    comp.detectionPrompt,
                    comp.avoidancePrompt,
                    comp.positvePrompt,
                    comp.negativePrompt,
                    comp.tab_index,
                    comp.image,
                    comp.image_batch,
                    comp.keep_original_filenames,
                    comp.input_batch_dir,
                    comp.output_batch_dir,
                    comp.keep_original_filenames_from_dir,
                    comp.show_batch_dir_results,
                    comp.input_video,
                    comp.video_output_dir,
                    comp.target_video_fps,
                    comp.upscaler_for_img2img,
                    comp.seed,
                    comp.sampler,
                    comp.scheduler,
                    comp.steps,
                    comp.box_threshold,
                    comp.mask_expand,
                    comp.mask_blur,
                    comp.max_resolution_on_detection,
                    comp.sam_model_name,
                    comp.dino_model_name,
                    comp.cfg_scale,
                    comp.denoise,
                    comp.inpaint_padding,
                    comp.inpainting_fill,
                    comp.width,
                    comp.height,
                    comp.batch_count,
                    comp.batch_size,
                    comp.inpainting_mask_invert,
                    comp.extra_includes,
                    comp.fix_steps,
                    comp.override_sd_model,
                    comp.sd_model_checkpoint,
                    comp.mask_num,
                    comp.avoid_mask_mode,
                    comp.avoidance_mask,
                    comp.only_custom_mask,
                    comp.custom_mask_mode,
                    comp.custom_mask,
                    comp.use_inpaint_diff,
                    comp.inpaint_diff_mask_view,
                    comp.lama_cleaner_upscaler,
                    comp.clip_skip,
                    comp.pass_into_hires_fix_automatically,
                    comp.save_before_hires_fix,
                    comp.previous_frame_into_controlnet,
                    comp.do_not_use_mask,

                    comp.hf_upscaler,
                    comp.hf_steps,
                    comp.hf_sampler,
                    comp.hf_scheduler,
                    comp.hf_denoise,
                    comp.hf_cfg_scale,
                    comp.hfPositivePromptSuffix,
                    comp.hf_size_limit,
                    comp.hf_above_limit_upscaler,
                    comp.hf_unload_detection_models,
                    comp.hf_disable_cn,
                    comp.hf_extra_mask_expand,
                    comp.hf_positvePrompt,
                    comp.hf_negativePrompt,
                    comp.hf_sd_model_checkpoint,
                    comp.hf_extra_inpaint_padding,
                    comp.hf_extra_mask_blur,
                    comp.hf_randomize_seed,
                    comp.hf_soft_inpaint,
                ] + comp.cn_inputs
                  + comp.soft_inpaint_inputs,
                outputs=[
                    comp.replacer_gallery,
                    comp.generation_info,
                    comp.html_info,
                    comp.html_log,
                ],
                show_progress=ui_toprow is None,
            )


            comp.apply_hires_fix_button.click(
                _js=getSubmitJsFunction('replacer', 'replacer_hf', 'replacer', True),
                fn=wrap_gradio_gpu_call(applyHiresFix, extra_outputs=[None, '', '']),
                inputs=[
                    comp.dummy_component, # task_id
                    comp.dummy_component, # gallery_idx
                    comp.replacer_gallery,
                    comp.generation_info,
                    comp.hf_upscaler,
                    comp.hf_steps,
                    comp.hf_sampler,
                    comp.hf_scheduler,
                    comp.hf_denoise,
                    comp.hf_cfg_scale,
                    comp.hfPositivePromptSuffix,
                    comp.hf_size_limit,
                    comp.hf_above_limit_upscaler,
                    comp.hf_unload_detection_models,
                    comp.hf_disable_cn,
                    comp.hf_extra_mask_expand,
                    comp.hf_positvePrompt,
                    comp.hf_negativePrompt,
                    comp.hf_sd_model_checkpoint,
                    comp.hf_extra_inpaint_padding,
                    comp.hf_extra_mask_blur,
                    comp.hf_randomize_seed,
                    comp.hf_soft_inpaint,
                ],
                outputs=[
                    comp.replacer_gallery,
                    comp.generation_info,
                    comp.html_info,
                    comp.html_log,
                ],
                show_progress=ui_toprow is None,
            )


            comp.random_seed.click(
                fn=lambda: -1,
                inputs=[
                ],
                outputs=[
                    comp.seed,
                ]
            )

            comp.reuse_seed.click(
                fn=getLastUsedSeed,
                inputs=[],
                outputs=[
                    comp.seed,
                ]
            )


            OuputPanelWatcher.send_back_to_replacer.click(
                fn=sendBackToReplacer,
                _js="sendBackToReplacer",
                inputs=[
                    comp.replacer_gallery,
                    comp.dummy_component
                ],
                outputs=[
                    comp.image,
                ]
            )

            if not doNotShowUnloadButton():
                comp.unload.click(
                    fn=wrap_queued_call(unloadModels),
                    inputs=[],
                    outputs=[])

            comp.avoid_mask_brush_color.change(
                fn=update_mask_brush_color,
                inputs=[comp.avoid_mask_brush_color],
                outputs=[comp.avoidance_mask]
            )

            comp.avoid_mask_create_canvas.click(
                fn=get_current_image,
                _js='replacerGetCurrentSourceImg',
                inputs=[
                    comp.dummy_component,
                    comp.trueComponent,
                    comp.avoid_mask_need_limit,
                    comp.max_resolution_on_detection
                ],
                outputs=[comp.avoidance_mask],
                postprocess=False,
            )


            comp.custom_mask_brush_color.change(
                fn=update_mask_brush_color,
                inputs=[comp.custom_mask_brush_color],
                outputs=[comp.custom_mask]
            )

            comp.create_canvas_custom_mask.click(
                fn=get_current_image,
                _js='replacerGetCurrentSourceImg',
                inputs=[
                    comp.dummy_component,
                    comp.falseComponent,
                    comp.custom_mask_need_limit,
                    comp.max_resolution_on_detection
                ],
                outputs=[comp.custom_mask],
                postprocess=False,
            )

            
            if replacer_extensions.InpaintDifferenceGlobals:
                comp.inpaint_diff_create.click(
                    fn=replacer_extensions.computeInpaintDifference,
                    inputs=[
                        comp.non_altered_image_for_inpaint_diff,
                        comp.image,
                        comp.mask_blur,
                        comp.inpaint_diff_mask_expand,
                        comp.inpaint_diff_mask_erosion,
                        comp.inpaint_diff_threshold,
                        comp.inpaint_diff_contours_only,
                    ],
                    outputs=[comp.inpaint_diff_mask_view],
                )

        self.components = comp


    def getReplacerTabUI(self):
        return self.replacerTabUI


replacerMainUI: ReplacerMainUI = None
replacerMainUI_dedicated: ReplacerMainUI = None

def initMainUI(*args):
    global replacerMainUI, replacerMainUI_dedicated
    try:
        replacer_extensions.initAllScripts()
        replacerMainUI = ReplacerMainUI(isDedicatedPage=False)
        replacerMainUI_dedicated = ReplacerMainUI(isDedicatedPage=True)
    finally:
        replacer_extensions.restoreTemporartChangedThigs()
