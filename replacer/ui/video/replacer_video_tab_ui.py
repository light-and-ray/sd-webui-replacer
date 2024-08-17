import gradio as gr
from modules import ui_settings, errors
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.ui_common import create_output_panel, update_generation_info
from replacer.ui.generate_ui import generate_ui, getLastUsedSeed, getLastUsedVariationSeed, getLastUsedMaskNum
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
    getSubmitJsFunction, sendBackToReplacer, IS_WEBUI_1_8, OutputPanelWatcher, ui_toprow,
    OverrideCustomScriptSource,
)
from replacer.tools import Pause
from replacer.ui.video.video_options_ui import makeVideoOptionsUI
from replacer.ui.video.video_project_ui import makeVideoProjectUI
from replacer.ui.video.video_masking_ui import makeVideoMaskingUI





def getVideoTabUI(isDedicatedPage: bool):
    comp = AttrDict()
    with OverrideCustomScriptSource('Video'):
        comp.selected_project_status = gr.Markdown("❌ Project is not selected")
        comp.selected_project = gr.Textbox(visible=False)
        with gr.Blocks(analytics_enabled=False) as replacerVideoTabUI:
            with gr.Tabs():
                with gr.Tab("Step 1 (Project)"):
                    makeVideoProjectUI(comp)
                with gr.Tab("Step 2 (Options)"):
                    makeVideoOptionsUI(comp)
                with gr.Tab("Step 3 (Masking)", elem_id="replacer_video_masking_tab"):
                    makeVideoMaskingUI(comp)
                with gr.Tab("Step 4 (First frame)"):
                    pass
                with gr.Tab("Step 5 (Generation)"):
                    pass
            comp.selected_project_status.render()
            comp.selected_project.render()

    return replacerVideoTabUI
