import gradio as gr
import modules
from modules import shared, sd_samplers, ui, ui_settings, errors
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.ui_components import ToolButton
from modules.ui_common import create_output_panel, refresh_symbol, update_generation_info, create_refresh_button
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from replacer.generate_ui import generate_ui, getLastUsedSeed
from replacer.apply_hires_fix import applyHiresFix
from replacer.options import (EXT_NAME, EXT_NAME_LOWER, getSaveDir, getDetectionPromptExamples,
    getPositivePromptExamples, getNegativePromptExamples, useFirstPositivePromptFromExamples,
    useFirstNegativePromptFromExamples, getHiresFixPositivePromptSuffixExamples,
    doNotShowUnloadButton, getAvoidancePromptExamples, getDedicatedPagePath,
    getDetectionPromptExamplesNumber, getAvoidancePromptExamplesNumber,
    getPositivePromptExamplesNumber, getNegativePromptExamplesNumber, getMaskColorStr
)
from replacer import replacer_scripts
from replacer.tools import limitSizeByOneDemention, OuputPanelWatcher, IS_WEBUI_1_9
from replacer.generate_ui import generate_ui


if IS_WEBUI_1_9:
    from modules import sd_schedulers

try:
    from modules import ui_toprow
except:
    ui_toprow = None

try:
    from modules.ui_components import ResizeHandleRow
except:
    ResizeHandleRow = gr.Row

try:
    from modules.ui_common import OutputPanel # webui 1.8+
    IS_WEBUI_1_8 = True
except Exception as e:
    IS_WEBUI_1_8 = False

if IS_WEBUI_1_8:
    from modules import infotext_utils

IS_WEBUI_1_5 = False
if not hasattr(sd_samplers, 'visible_sampler_names'): # webui 1.5
    sd_samplers.visible_sampler_names = lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in shared.opts.hide_samplers]
    IS_WEBUI_1_5 = True


def getHiresFixCheckpoints():
    if IS_WEBUI_1_5:
        return ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles()
    else:
        return ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=False)



def update_mask_brush_color(color):
    return gr.Image.update(brush_color=color)

def get_current_image(image, isAvoid, needLimit, maxResolutionOnDetection):
    if image is None:
        return
    if needLimit:
        image = decode_base64_to_image(image)
        image = limitSizeByOneDemention(image, maxResolutionOnDetection)
        image = 'data:image/png;base64,' + encode_pil_to_base64(image).decode()
    return gr.Image.update(image)


def unloadModels():
    mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
    memBefore = mem_stats['reserved']
    from scripts.sam import clear_cache
    clear_cache()
    mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
    memAfter = mem_stats['reserved']
    
    text = f'[{EXT_NAME}] {(memBefore - memAfter) / 1024 :.2f} GB of VRAM were freed'
    print(text, flush=True)
    if not IS_WEBUI_1_5:
        gr.Info(text)



def getSubmitJsFunction(galleryId, buttonsId, extraShowButtonsId, fillGalleryIdx):
    if not ui_toprow:
        return ''
    fillGalleryIdxCode = ''
    if fillGalleryIdx:
        fillGalleryIdxCode = 'arguments_[1] = selected_gallery_index();'
    return 'function(){'\
        'var arguments_ = Array.from(arguments);'\
        f'{fillGalleryIdxCode}'\
        f'arguments_.push("{extraShowButtonsId}", "{buttonsId}", "{galleryId}");'\
        'return submit_replacer.apply(null, arguments_);'\
    '}'


def sendBackToReplacer(gallery, gallery_index):
    assert len(gallery) > 0, 'No image'
    assert 0 <= gallery_index < len(gallery), f'Bad image index: {gallery_index}'
    assert not IS_WEBUI_1_5, 'sendBackToReplacer is not supported for webui < 1.8'
    image_info = gallery[gallery_index] if 0 <= gallery_index < len(gallery) else gallery[0]
    image = infotext_utils.image_from_url_text(image_info)
    return image


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class ReplacerMainUI:
    def __init__(self, isDedicatedPage):
        self.replacerTabUI = None
        self.components = AttrDict()
        self.init_tab(isDedicatedPage)

    def init_tab(self, isDedicatedPage):
        comp = AttrDict()
        with gr.Blocks(analytics_enabled=False) as self.replacerTabUI:
            comp.tab_index = gr.Number(value=0, visible=False)
            comp.dummy_component = gr.Label(visible=False)
            comp.trueComponent = gr.Checkbox(value=True, visible=False)
            comp.falseComponent = gr.Checkbox(value=False, visible=False)
            if replacer_scripts.script_controlnet:
                try:
                    cnUiGroupsLenBefore = len(replacer_scripts.ControlNetUiGroup.all_ui_groups)
                except Exception as e:
                    errors.report(f"Cannot init cnUiGroupsLenBefore: {e}", exc_info=True)
                    replacer_scripts.script_controlnet = None

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


                    with gr.Accordion("Advanced options", open=False, elem_id='replacer_advanced_options'):
                        with gr.Tabs(elem_id="replacer_advanced_options_tabs"):
                            with gr.Tab('Generation'):
                                with gr.Row():
                                    sampler_names = sd_samplers.visible_sampler_names()
                                    defaultSampler = "DPM++ 2M SDE" if IS_WEBUI_1_9 else "DPM++ 2M SDE Karras"
                                    comp.sampler = gr.Dropdown(
                                        label='Sampling method',
                                        elem_id="replacer_sampler",
                                        choices=sampler_names,
                                        value=defaultSampler
                                    )

                                    if IS_WEBUI_1_9:
                                        scheduler_names = [x.label for x in sd_schedulers.schedulers]
                                        comp.scheduler = gr.Dropdown(
                                            label='Schedule type',
                                            elem_id=f"replacer_scheduler",
                                            choices=scheduler_names,
                                            value=scheduler_names[0])
                                    else:
                                        comp.scheduler = gr.Textbox("", visible=False)

                                    comp.steps = gr.Slider(
                                        label='Steps',
                                        value=20,
                                        step=1,
                                        minimum=1,
                                        maximum=150,
                                        elem_id="replacer_steps"
                                    )
                                
                                with gr.Row():
                                    comp.cfg_scale = gr.Slider(label='CFG Scale',
                                        value=5.5, elem_id="replacer_cfg_scale",
                                        minimum=1.0, maximum=30.0, step=0.5)

                                    comp.fix_steps = gr.Checkbox(label='Do exactly the amount of steps the slider specifies',
                                        value=False, elem_id="replacer_fix_steps")

                                with gr.Row():
                                    with gr.Column(elem_id="replacer_width_height_column", elem_classes="replacer-generation-size"):
                                        comp.width = gr.Slider(label='width',
                                            value=512, elem_id="replacer_width",
                                            minimum=64, maximum=2048, step=8)
                                        comp.height = gr.Slider(label='height',
                                            value=512, elem_id="replacer_height",
                                            minimum=64, maximum=2048, step=8)
                                    with gr.Column(elem_id="replacer_batch_count_size_column", elem_classes="replacer-batch-count-size"):
                                        comp.batch_count = gr.Slider(label='batch count',
                                            value=1, elem_id="replacer_batch_count",
                                            minimum=1, maximum=10, step=1)
                                        comp.batch_size = gr.Slider(label='batch size',
                                            value=1, elem_id="replacer_batch_size",
                                            minimum=1, maximum=10, step=1)

                                with gr.Row():
                                    comp.upscaler_for_img2img = gr.Dropdown(
                                        value="None",
                                        choices=[x.name for x in shared.sd_upscalers],
                                        label="Upscaler for img2Img",
                                        elem_id="replacer_upscaler_for_img2img",
                                    )

                                    if shared.cmd_opts.use_textbox_seed:
                                        comp.seed = gr.Textbox(label='Seed', value="", elem_id="replacer_seed", min_width=100)
                                    else:
                                        comp.seed = gr.Number(label='Seed', value=-1, elem_id="replacer_seed", min_width=100, precision=0)

                                    comp.random_seed = ToolButton(
                                        ui.random_symbol,
                                        elem_id="replacer_random_seed",
                                        label='Random seed'
                                    )
                                    comp.reuse_seed = ToolButton(
                                        ui.reuse_symbol,
                                        elem_id="replacer_reuse_seed",
                                        label='Reuse seed'
                                    )
                                
                                with gr.Row():
                                    if not isDedicatedPage:
                                        comp.sd_model_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')
                                        comp.override_sd_model = gr.Checkbox(label='Override stable diffusion model',
                                            value=False, elem_id="replacer_override_sd_model")

                                    comp.clip_skip = ui_settings.create_setting_component('CLIP_stop_at_last_layers')


                            with gr.Tab('Detection'):
                                with gr.Row():
                                    comp.box_threshold = gr.Slider(label='Box Threshold',
                                        value=0.3, elem_id="replacer_box_threshold",
                                        minimum=0.0, maximum=1.0, step=0.05)
                                    comp.mask_expand = gr.Slider(label='Mask Expand',
                                        value=35, elem_id="replacer_mask_expand",
                                        minimum=-50, maximum=100, step=1)

                                with gr.Row():
                                    if not doNotShowUnloadButton():
                                        comp.unload = gr.Button(
                                            value="Unload detection models",
                                            elem_id="replacer_unload_detection_models")

                                    comp.max_resolution_on_detection = gr.Slider(
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

                                    comp.sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list,
                                        value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                                    comp.sam_refresh_models = ToolButton(value=refresh_symbol)
                                    comp.sam_refresh_models.click(refresh_sam_models, comp.sam_model_name, comp.sam_model_name)

                                    comp.dino_model_name = gr.Dropdown(label="GroundingDINO Model", choices=dino_model_list, value=dino_model_list[0])
                                
                                with gr.Row():
                                    comp.mask_num = gr.Radio(label='Mask num',
                                        choices=['Random', '1', '2', '3'],
                                        value='Random', type="value", elem_id="replacer_mask_num")

                                with gr.Row():
                                    comp.extra_includes = ui_settings.create_setting_component(EXT_NAME_LOWER + "_default_extra_includes")
                                    comp.extra_includes.label = 'Extra include in gallery'

                            with gr.Tab('Inpainting'):
                                with gr.Row():
                                    comp.mask_blur = gr.Slider(label='Mask Blur',
                                        value=4, elem_id="replacer_mask_blur",
                                        minimum=0, maximum=100, step=1)
                                    comp.inpaint_padding = gr.Slider(label='Padding',
                                        value=40, elem_id="replacer_inpaint_padding",
                                        minimum=0, maximum=1000, step=1)

                                with gr.Row():
                                    comp.denoise = gr.Slider(label='Denoising',
                                        value=1.0, elem_id="replacer_denoise",
                                        minimum=0.0, maximum=1.0, step=0.01)

                                with gr.Row():
                                    comp.inpainting_fill = gr.Radio(label='Masked content',
                                        choices=['fill', 'original', 'latent noise', 'latent nothing'],
                                        value='fill', type="index", elem_id="replacer_inpainting_fill")

                                with gr.Row():
                                    if replacer_scripts.script_lama_cleaner_as_masked_content:
                                        comp.lama_cleaner_upscaler = ui_settings.create_setting_component('upscaling_upscaler_for_lama_cleaner_masked_content')
                                    else:
                                        comp.lama_cleaner_upscaler = gr.Textbox(visible=False)

                                with gr.Row():
                                    comp.inpainting_mask_invert = gr.Radio(
                                        label='Mask mode',
                                        choices=['Inpaint masked', 'Inpaint not masked'],
                                        value='Inpaint masked',
                                        type="index",
                                        elem_id="replacer_mask_mode")

                                comp.soft_inpaint_inputs = []
                                if replacer_scripts.script_soft_inpaint:
                                    try:
                                        with gr.Row():
                                            replacer_scripts.needWatchSoftInpaintUI = True
                                            comp.soft_inpaint_inputs = list(replacer_scripts.script_soft_inpaint.ui(True))
                                            replacer_scripts.needWatchSoftInpaintUI = False
                                            from modules.ui_components import InputAccordion
                                            new_soft_inpaint_accordion = InputAccordion(False, label="Soft inpainting", elem_id="replaer_soft_inpainting_enabled")
                                            new_soft_inpaint_accordion.accordion.children = soft_inpaint_inputs[0].accordion.children
                                            for child in new_soft_inpaint_accordion.accordion.children:
                                                child.parent = new_soft_inpaint_accordion.accordion
                                            comp.soft_inpaint_inputs[0].accordion.visible = False
                                            comp.soft_inpaint_inputs[0] = new_soft_inpaint_accordion
                                    except Exception as e:
                                        errors.report(f"Cannot add soft inpaint accordion {e}", exc_info=True)
                                        replacer_scripts.script_soft_inpaint = None


                            with gr.Tab('Avoidance'):
                                with gr.Row():
                                    comp.avoidancePrompt = gr.Textbox(label="Avoidance prompt",
                                                        show_label=True,
                                                        lines=1,
                                                        elem_classes=["avoidancePrompt"],
                                                        placeholder=None,
                                                        elem_id="replacer_avoidancePrompt")

                                    gr.Examples(
                                        examples=getAvoidancePromptExamples(),
                                        inputs=comp.avoidancePrompt,
                                        label="",
                                        elem_id="replacer_avoidancePrompt_examples",
                                        examples_per_page=getAvoidancePromptExamplesNumber(),
                                    )

                                with gr.Row():
                                    comp.avoid_mask_create_canvas = gr.Button('Create canvas', elem_id='replacer_avoid_mask_create_canvas')
                                    comp.avoid_mask_need_limit = gr.Checkbox(value=True, label='Limit avoidance mask canvas resolution on creating')
                                    comp.avoid_mask_mode = gr.CheckboxGroup(['Draw mask', 'Upload mask'], value=['Draw mask'], label="Canvas mask source")
                                with gr.Row():
                                    comp.avoidance_mask = gr.Image(
                                        label="Avoidance mask",
                                        show_label=False,
                                        elem_id="replacer_avoidance_mask",
                                        source="upload",
                                        interactive=True,
                                        type="pil",
                                        tool="sketch",
                                        image_mode="RGBA",
                                        brush_color=getMaskColorStr()
                                    )
                                with gr.Row():
                                    comp.avoid_mask_brush_color = gr.ColorPicker(
                                        getMaskColorStr(), label='Brush color',
                                        info='visual only, use when brush color is hard to see'
                                    )
                                    if IS_WEBUI_1_5:
                                        comp.avoid_mask_brush_color.visible = False
                            
                            with gr.Tab('Custom mask'):
                                with gr.Row():
                                    comp.only_custom_mask = gr.Checkbox(label='Do not use detection prompt if use custom mask',
                                        value=True, elem_id="replacer_only_custom_mask")

                                with gr.Row():
                                    comp.create_canvas_custom_mask = gr.Button('Create canvas', elem_id='replacer_create_canvas_custom_mask')
                                    comp.custom_mask_need_limit = gr.Checkbox(value=True, label='Limit custom mask canvas resolution on creating')
                                    comp.custom_mask_mode = gr.CheckboxGroup(['Draw mask', 'Upload mask'], value=['Draw mask'], label="Canvas mask source")
                                with gr.Row():
                                    comp.custom_mask = gr.Image(
                                        label="Custom mask",
                                        show_label=False,
                                        elem_id="replacer_custom_mask",
                                        source="upload",
                                        interactive=True,
                                        type="pil",
                                        tool="sketch",
                                        image_mode="RGBA",
                                        brush_color=getMaskColorStr()
                                    )
                                with gr.Row():
                                    comp.custom_mask_brush_color = gr.ColorPicker(
                                        getMaskColorStr(), label='Brush color',
                                        info='visual only, use when brush color is hard to see')
                                    if IS_WEBUI_1_5:
                                        comp.custom_mask_brush_color.visible = False

                            with (gr.Tab('Inpaint Diff') if replacer_scripts.InpaintDifferenceGlobals
                                    else gr.Group()) as comp.inpaint_diff_tab:
                                with gr.Row():
                                    comp.inpaint_diff_create = gr.Button('Create', elem_id='replacer_inpaint_diff_create')
                                    comp.use_inpaint_diff = gr.Checkbox(label='Use inpaint difference',
                                        value=True, elem_id="replacer_use_inpaint_diff")
                                with gr.Row():
                                    comp.non_altered_image_for_inpaint_diff = gr.Image(
                                        label="Non altered image",
                                        show_label=True,
                                        elem_id="replacer_non_altered_image_for_inpaint_diff",
                                        source="upload",
                                        type="pil",
                                        image_mode="RGBA",
                                    )
                                    comp.inpaint_diff_mask_view = gr.Image(label="Difference mask",
                                        interactive=True, type="pil",
                                        elem_id="replacer_inpaint_diff_mask_view")
                                with gr.Row():
                                    comp.inpaint_diff_threshold = gr.Slider(label='Difference threshold',
                                        maximum=1, step=0.01, value=1, elem_id='inpaint_difference_difference_threshold')
                                    comp.inpaint_diff_mask_expand = gr.Slider(label='Mask dilation',
                                        value=5, elem_id="replacer_inpaint_diff_mask_expand",
                                        minimum=0, maximum=100, step=1)
                                    comp.inpaint_diff_mask_erosion = gr.Slider(label='Mask erosion',
                                        maximum=100, step=1, value=0, elem_id='inpaint_difference_mask_erosion')
                                with gr.Row():
                                    comp.inpaint_diff_contours_only = gr.Checkbox(label='Contours only',
                                        value=False, elem_id='inpaint_difference_contours_only')
                            if not replacer_scripts.InpaintDifferenceGlobals:
                                comp.inpaint_diff_tab.visible = False
                                comp.inpaint_diff_tab.render = False


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
                            gr.Markdown("To increase consistency it's better to inpaint clear "\
                                "objects on video with good quality and enough context. "\
                                "Your prompts need to produce consistent results\n\n"\
                                "To suppress flickering you can generate in little fps (e.g. 10), "\
                                "then interpolate (x2) it with ai interpolation algorithm "\
                                "(e.g [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) or "\
                                "[frame interpolation in deforum sd-webui extension]("\
                                "https://github.com/deforum-art/sd-webui-deforum/wiki/Upscaling-and-Frame-Interpolation))\n\n"\
                                "You can also use [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) or "\
                                "[lama-cleaner](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content) with (low denosing) "\
                                "extensions to increase consistency, if it fits to your scenario")
                    
                    comp.cn_inputs = []
                    if replacer_scripts.script_controlnet:
                        try:
                            with gr.Row():
                                replacer_scripts.needWatchControlNetUI = True
                                comp.cn_inputs = list(replacer_scripts.script_controlnet.ui(True))
                                replacer_scripts.needWatchControlNetUI = False

                                if not replacer_scripts.controlNetAccordion:
                                    errors.report(f"[{EXT_NAME}] controlnet accordion wasn't found", exc_info=True)
                                    replacer_scripts.script_controlnet = None
                                else:
                                    with replacer_scripts.controlNetAccordion:
                                        with gr.Row():
                                            gr.Markdown('_If you select Inpaint -> inpaint_only, cn inpaint model will be used instead of sd inpainting_')
                        except Exception as e:
                            errors.report(f"Cannot add controlnet accordion {e}", exc_info=True)
                            replacer_scripts.script_controlnet = None


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
                        with gr.Accordion("HiresFix options", open=False):
                            with gr.Tabs():
                                with gr.Tab('General'):
                                    with gr.Row():
                                        comp.hf_upscaler = gr.Dropdown(
                                            value="ESRGAN_4x",
                                            choices=[x.name for x in shared.sd_upscalers],
                                            label="Upscaler",
                                        )

                                        comp.hf_steps = gr.Slider(
                                            label='Hires steps',
                                            value=4,
                                            step=1,
                                            minimum=0,
                                            maximum=150,
                                            elem_id="replacer_hf_steps"
                                        )

                                    with gr.Row():
                                        comp.hf_denoise = gr.Slider(
                                            label='Hires Denoising',
                                            value=0.35,
                                            step=0.01,
                                            minimum=0.0,
                                            maximum=1.0,
                                            elem_id="replacer_hf_denoise",
                                        )

                                    with gr.Row():
                                        comp.hf_size_limit = gr.Slider(
                                            label='Limit render size',
                                            value=1800,
                                            step=1,
                                            minimum=700,
                                            maximum=10000,
                                            elem_id="replacer_hf_size_limit",
                                        )

                                        comp.hf_above_limit_upscaler = gr.Dropdown(
                                            value="Lanczos",
                                            choices=[x.name for x in shared.sd_upscalers],
                                            label="Above limit upscaler",
                                        )
                                    
                                    with gr.Row():
                                        comp.hf_extra_mask_expand = gr.Slider(
                                            label='Extra mask expand',
                                            value=5,
                                            step=1,
                                            minimum=0,
                                            maximum=200,
                                            elem_id="replacer_hf_extra_mask_expand",
                                        )

                                        comp.hf_extra_inpaint_padding = gr.Slider(label='Extra inpaint padding',
                                            value=250, elem_id="replacer_hf_extra_inpaint_padding",
                                            minimum=0, maximum=3000, step=1)
                                        
                                        comp.hf_extra_mask_blur = gr.Slider(label='Extra mask blur',
                                            value=2, elem_id="replacer_hf_extra_mask_blur",
                                            minimum=0, maximum=150, step=1)

                                    with gr.Row():
                                        comp.hf_randomize_seed = gr.Checkbox(
                                            label='Randomize seed for hires fix',
                                            value=True,
                                            elem_id="replacer_hf_randomize_seed",
                                        )

                                with gr.Tab('Advanced'):
                                    with gr.Row():
                                        comp.hf_sampler = gr.Dropdown(
                                            label='Hires sampling method',
                                            elem_id="replacer_hf_sampler",
                                            choices=["Use same sampler"] + sd_samplers.visible_sampler_names(),
                                            value="Use same sampler"
                                        )
                                        if IS_WEBUI_1_9:
                                            comp.hf_scheduler = gr.Dropdown(
                                                label='Hires schedule type',
                                                elem_id="replacer_hf_scheduler",
                                                choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers],
                                                value="Use same scheduler"
                                            )
                                        else:
                                            comp.hf_scheduler = gr.Textbox("", visible=False)

                                        comp.hf_cfg_scale = gr.Slider(
                                            label='Hires CFG Scale',
                                            value=1.0,
                                            step=0.5,
                                            minimum=1.0,
                                            maximum=30.0,
                                            elem_id="replacer_hf_cfg_scale"
                                        )

                                    with gr.Row():
                                        comp.hf_unload_detection_models = gr.Checkbox(
                                            label='Unload detection models before hires fix',
                                            value=True,
                                            elem_id="replacer_hf_unload_detection_models",
                                        )
                                        if doNotShowUnloadButton():
                                            comp.hf_unload_detection_models.visible = False

                                    with gr.Row():
                                        placeholder = None
                                        placeholder = getHiresFixPositivePromptSuffixExamples()[0]

                                        comp.hfPositivePromptSuffix = gr.Textbox(
                                            label="Suffix for positive prompt",
                                            show_label=True,
                                            lines=1,
                                            elem_classes=["hfPositivePromptSuffix"],
                                            placeholder=placeholder,
                                            elem_id="replacer_hfPositivePromptSuffix",
                                        )

                                        gr.Examples(
                                            examples=getHiresFixPositivePromptSuffixExamples(),
                                            inputs=comp.hfPositivePromptSuffix,
                                            label="",
                                            elem_id="replacer_hfPositivePromptSuffix_examples",
                                        )

                                    with gr.Row():
                                        comp.hf_positvePrompt = gr.Textbox(label="Override positive prompt",
                                                show_label=True,
                                                lines=1,
                                                elem_classes=["positvePrompt"],
                                                placeholder='leave empty to use the same prompt',
                                                elem_id="replacer_hf_positvePrompt")

                                        comp.hf_negativePrompt = gr.Textbox(label="Override negative prompt",
                                                show_label=True,
                                                lines=1,
                                                elem_classes=["negativePrompt"],
                                                placeholder='leave empty to use the same prompt',
                                                elem_id="replacer_hf_negativePrompt")

                                    with gr.Row():
                                        comp.hf_sd_model_checkpoint = gr.Dropdown(label='Hires checkpoint',
                                            elem_id="replacer_hf_sd_model_checkpoint",
                                            choices=getHiresFixCheckpoints(), value="Use same checkpoint")
                                        create_refresh_button(comp.hf_sd_model_checkpoint, modules.sd_models.list_models,
                                            lambda: {"choices": getHiresFixCheckpoints()}, "replacer_hf_sd_model_checkpoint")
                                        
                                        comp.hf_disable_cn = gr.Checkbox(
                                            label='Disable ControlNet while hires fix',
                                            value=True,
                                            elem_id="replacer_hf_disable_cn",
                                        )
                                        if not replacer_scripts.script_controlnet:
                                            comp.hf_disable_cn.visible = False
                                    
                                    with gr.Row():
                                        comp.hf_soft_inpaint = gr.Radio(label='Soft inpainting for hires fix',
                                            choices=['Same', 'Enable', 'Disable'],
                                            value='Same', type="value", elem_id="replacer_hf_soft_inpaint")
                                        if not replacer_scripts.script_soft_inpaint:
                                            comp.hf_soft_inpaint.visible = False

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



            if replacer_scripts.script_controlnet:
                try:
                    replacer_scripts.ControlNetUiGroup.a1111_context.img2img_w_slider = comp.width
                    replacer_scripts.ControlNetUiGroup.a1111_context.img2img_h_slider = comp.height

                    for ui_group in replacer_scripts.ControlNetUiGroup.all_ui_groups[cnUiGroupsLenBefore:]:
                        ui_group.register_run_annotator()
                        if not replacer_scripts.IS_SD_WEBUI_FORGE:
                            ui_group.inpaint_crop_input_image.value = True
                            ui_group.inpaint_crop_input_image.visible = True
                            ui_group.inpaint_crop_input_image.label = "Crop input image based on generated mask",
                        # if isDedicatedPage: 
                        #     replacer_scripts.ControlNetUiGroup.a1111_context.setting_sd_model_checkpoint = sd_model_checkpoint
                        # ui_group.register_sd_version_changed()
                except Exception as e:
                    errors.report(f"Cannot change ControlNet accordion entry: {e}", exc_info=True)
                    replacer_scripts.script_controlnet = None


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

            
            if replacer_scripts.InpaintDifferenceGlobals:
                comp.inpaint_diff_create.click(
                    fn=replacer_scripts.computeInpaintDifference,
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
    replacerMainUI = ReplacerMainUI(isDedicatedPage=False)
    replacerMainUI_dedicated = ReplacerMainUI(isDedicatedPage=True)
