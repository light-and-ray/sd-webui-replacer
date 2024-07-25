import gradio as gr
from modules import shared, ui_settings, errors, infotext_utils
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.ui_common import create_output_panel, update_generation_info
from replacer.ui.generate_ui import generate_ui, getLastUsedSeed, getLastUsedVariationSeed
from replacer.ui.apply_hires_fix import applyHiresFix
from replacer.options import (EXT_NAME, getSaveDir, getDetectionPromptExamples,
    getPositivePromptExamples, getNegativePromptExamples, useFirstPositivePromptFromExamples,
    useFirstNegativePromptFromExamples, doNotShowUnloadButton, getDedicatedPagePath,
    getDetectionPromptExamplesNumber, getPositivePromptExamplesNumber, getNegativePromptExamplesNumber
)
from replacer.extensions import replacer_extensions
from replacer.ui.make_advanced_options import makeAdvancedOptions
from replacer.ui.make_hiresfix_options import makeHiresFixOptions
from replacer.ui.video_ui import makeVideoUI
from replacer.ui.tools_ui import ( update_mask_brush_color, get_current_image, unloadModels, AttrDict,
    getSubmitJsFunction, sendBackToReplacer, IS_WEBUI_1_8, OutputPanelWatcher, ui_toprow,
    OverrideCustomScriptSource,
)
from replacer.tools import Pause


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
            comp.selected_input_mode = gr.Textbox(value="tab_single", visible=False)
            comp.dummy_component = gr.Label(visible=False)
            comp.trueComponent = gr.Checkbox(value=True, visible=False)
            comp.falseComponent = gr.Checkbox(value=False, visible=False)
            if replacer_extensions.controlnet.SCRIPT:
                try:
                    cnUiGroupsLenBefore = len(replacer_extensions.controlnet.ControlNetUiGroup.all_ui_groups)
                except Exception as e:
                    errors.report(f"Cannot init cnUiGroupsLenBefore: {e}", exc_info=True)
                    replacer_extensions.controlnet.SCRIPT = None

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

                        if getDetectionPromptExamplesNumber() > 0:
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

                        comp.positivePrompt = gr.Textbox(label="Positive prompt",
                                            show_label=True,
                                            lines=1,
                                            elem_classes=["positivePrompt"],
                                            placeholder=placeholder,
                                            elem_id="replacer_positivePrompt")

                        if getPositivePromptExamplesNumber() > 0:
                            gr.Examples(
                                examples=getPositivePromptExamples(),
                                inputs=comp.positivePrompt,
                                label="",
                                elem_id="replacer_positivePrompt_examples",
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

                        if getNegativePromptExamplesNumber() > 0:
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
                            comp.image_batch = gr.Files(label="Files", interactive=True, elem_id="replacer_image_batch")
                            comp.keep_original_filenames = gr.Checkbox(
                                label='Keep original filenames', value=True, elem_id="replacer_keep_original_filenames")

                        with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="replacer_batch_directory_tab") as comp.tab_batch_dir:
                            comp.input_batch_dir = gr.Textbox(
                                label="Input directory",
                                placeholder="A directory on the same machine where the server is running.",
                                elem_id="replacer_input_batch_dir")
                            comp.output_batch_dir = gr.Textbox(
                                label="Output directory",
                                placeholder="Leave blank to save images to the default path.",
                                elem_id="replacer_output_batch_dir")
                            comp.keep_original_filenames_from_dir = gr.Checkbox(
                                label='Keep original filenames (batch from dir)', value=True, elem_id="replacer_keep_original_filenames_from_dir")
                            comp.show_batch_dir_results = gr.Checkbox(
                                label='Show result images', value=False, elem_id="replacer_show_batch_dir_results")

                        with gr.TabItem('Video', id="batch_from_video", elem_id="replacer_batch_video_tab") as comp.tab_batch_video:
                            makeVideoUI(comp)

                    comp.cn_inputs = []

                    with OverrideCustomScriptSource("controlnet"):
                        if replacer_extensions.controlnet.SCRIPT:
                            replacer_extensions.controlnet.ControlNetUiGroup.a1111_context.img2img_submit_button = comp.run_button
                            try:
                                with gr.Row():
                                    replacer_extensions.controlnet.needWatchControlNetUI = True
                                    comp.cn_inputs = list(replacer_extensions.controlnet.SCRIPT.ui(True))
                                    replacer_extensions.controlnet.needWatchControlNetUI = False

                                    if not replacer_extensions.controlnet.controlNetAccordion:
                                        errors.report(f"[{EXT_NAME}] controlnet accordion wasn't found", exc_info=True)
                                    else:
                                        with replacer_extensions.controlnet.controlNetAccordion:
                                            with gr.Row():
                                                gr.Markdown('_If you select Inpaint -> inpaint_only, cn inpaint model will be used instead of sd inpainting_')
                            except Exception as e:
                                errors.report(f"Cannot add controlnet accordion {e}", exc_info=True)
                                replacer_extensions.controlnet.SCRIPT = None


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
                        if isDedicatedPage and OutputPanelWatcher.send_to_img2img:
                            OutputPanelWatcher.send_to_img2img.visible = False
                            OutputPanelWatcher.send_to_inpaint.visible = False
                            OutputPanelWatcher.send_to_extras.visible = False

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

                    comp.pause_button = gr.Button(
                        '-',
                        elem_id='replacer_pause',
                        visible=False,
                        elem_classes=["pause-button"],
                        variant='compact'
                    )

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



            if replacer_extensions.controlnet.SCRIPT:
                try:
                    replacer_extensions.controlnet.ControlNetUiGroup.a1111_context.img2img_w_slider = comp.width
                    replacer_extensions.controlnet.ControlNetUiGroup.a1111_context.img2img_h_slider = comp.height

                    for ui_group in replacer_extensions.controlnet.ControlNetUiGroup.all_ui_groups[cnUiGroupsLenBefore:]:
                        ui_group.register_run_annotator()
                        if not replacer_extensions.controlnet.IS_SD_WEBUI_FORGE:
                            ui_group.inpaint_crop_input_image.value = True
                            ui_group.inpaint_crop_input_image.visible = True
                            ui_group.inpaint_crop_input_image.label = "Crop input image based on generated mask",
                        # if isDedicatedPage:
                        #     replacer_extensions.controlnet.ControlNetUiGroup.a1111_context.setting_sd_model_checkpoint = sd_model_checkpoint
                        # ui_group.register_sd_version_changed()
                except Exception as e:
                    errors.report(f"Cannot change ControlNet accordion entry: {e}", exc_info=True)
                    replacer_extensions.controlnet.SCRIPT = None


            def tabSelected(tab, showPause, isVideo):
                def func():
                    if not isVideo:
                        text = 'batch'
                    else:
                        text = 'video'
                    return tab, gr.update(value=f'pause/resume {text} generation', visible=showPause)
                return func

            comp.tab_single.select(fn=tabSelected("tab_single", False, False), inputs=[], outputs=[comp.selected_input_mode, comp.pause_button])
            comp.tab_batch.select(fn=tabSelected("tab_batch", True, False), inputs=[], outputs=[comp.selected_input_mode, comp.pause_button])
            comp.tab_batch_dir.select(fn=tabSelected("tab_batch_dir", True, False), inputs=[], outputs=[comp.selected_input_mode, comp.pause_button])
            comp.tab_batch_video.select(fn=tabSelected("tab_batch_video", True, True), inputs=[], outputs=[comp.selected_input_mode, comp.pause_button])


            comp.run_button.click(
                _js=getSubmitJsFunction('replacer', 'replacer', 'replacer_hf', False),
                fn=wrap_gradio_gpu_call(generate_ui, extra_outputs=[None, '', '']),
                inputs=[
                    comp.dummy_component, # task_id
                    comp.selected_input_mode,
                    comp.detectionPrompt,
                    comp.avoidancePrompt,
                    comp.positivePrompt,
                    comp.negativePrompt,
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
                    comp.clip_skip,
                    comp.pass_into_hires_fix_automatically,
                    comp.save_before_hires_fix,
                    comp.previous_frame_into_controlnet,
                    comp.do_not_use_mask,
                    comp.selected_video_mode,
                    comp.rotation_fix,
                    comp.variation_seed,
                    comp.variation_strength,
                    comp.integer_only_masked,
                    comp.forbid_too_small_crop_region,

                    comp.ad_fragment_length,
                    comp.ad_internal_fps,
                    comp.ad_batch_size,
                    comp.ad_stride,
                    comp.ad_overlap,
                    comp.ad_latent_power,
                    comp.ad_latent_scale,
                    comp.ad_generate_only_first_fragment,
                    comp.ad_cn_inpainting_model,
                    comp.ad_control_weight,
                    comp.ad_force_override_sd_model,
                    comp.ad_force_sd_model_checkpoint,
                    comp.ad_motion_model,

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
                    comp.hf_positivePrompt,
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
                    comp.hf_positivePrompt,
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

            comp.random_variation_seed.click(
                fn=lambda: -1,
                inputs=[
                ],
                outputs=[
                    comp.variation_seed,
                ]
            )

            comp.reuse_variation_seed.click(
                fn=getLastUsedVariationSeed,
                inputs=[],
                outputs=[
                    comp.variation_seed,
                ]
            )


            OutputPanelWatcher.send_back_to_replacer.click(
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

            comp.pause_button.click(
                fn=Pause.toggle
            )


            if replacer_extensions.inpaint_difference.Globals:
                comp.inpaint_diff_create.click(
                    fn=replacer_extensions.inpaint_difference.computeInpaintDifference,
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

registered_param_bindings_main_ui = []

def initMainUI(*args):
    global replacerMainUI, replacerMainUI_dedicated, registered_param_bindings_main_ui
    lenBefore = len(infotext_utils.registered_param_bindings)
    try:
        replacer_extensions.initAllScripts()
        replacerMainUI = ReplacerMainUI(isDedicatedPage=False)
        replacerMainUI_dedicated = ReplacerMainUI(isDedicatedPage=True)
    finally:
        replacer_extensions.restoreTemporaryChangedThings()

    registered_param_bindings_main_ui = infotext_utils.registered_param_bindings[lenBefore:]


def reinitMainUIAfterUICreated():
    replacer_extensions.reinitAllScriptsAfterUICreated()
    infotext_utils.registered_param_bindings += registered_param_bindings_main_ui

