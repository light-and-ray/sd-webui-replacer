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




def getVideoTabUI(comp: AttrDict, isDedicatedPage: bool):
    with OverrideCustomScriptSource('Video'):
        with gr.Blocks(analytics_enabled=False) as replacerVideoTabUI:
            with gr.Tabs():
                with gr.Tab("Step 1 (Project)"):
                    makeVideoProjectUI(comp)
                with gr.Tab("Step 2 (Masking)"):
                    pass
                with gr.Tab("Step 3 (First frame)"):
                    pass
                with gr.Tab("Step 4 (Options)"):
                    makeVideoOptionsUI(comp)
                with gr.Tab("Step 5 (Generation)"):
                    pass
    return replacerVideoTabUI
