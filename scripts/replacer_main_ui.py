import copy
import gradio as gr
from modules import script_callbacks, progress, shared, errors, ui_postprocessing
from replacer.options import (EXT_NAME, EXT_NAME_LOWER, needHideSegmentAnythingAccordions,
    getDedicatedPagePath, on_ui_settings, needHideAnimateDiffAccordions, hideVideoInMainUI, extrasInDedicated,
)
from replacer.ui.tools_ui import IS_WEBUI_1_5
from replacer.ui import replacer_main_ui
from replacer.tools import getReplacerFooter
from replacer.ui.tools_ui import watchOutputPanel, watchSetCustomScriptSourceForComponents
from replacer.extensions import replacer_extensions



def on_ui_tabs():
    result = []
    replacer_main_ui.reinitMainUIAfterUICreated()
    tab = replacer_main_ui.replacerMainUI.getReplacerTabUI()
    result.append((tab, EXT_NAME, EXT_NAME))
    if not hideVideoInMainUI():
        video_tab = replacer_main_ui.replacerMainUI.getReplacerVideoTabUI()
        video_title = f"{EXT_NAME} - video"
        result.append((video_tab, video_title, video_title))
    return result

script_callbacks.on_ui_tabs(on_ui_tabs)


def mountDedicatedPage(demo, app):
    try:
        path = getDedicatedPagePath()
        app.add_api_route(f"{path}/internal/progress",
            progress.progressapi, methods=["POST"],
            response_model=progress.ProgressResponse)
        replacer_extensions.image_comparison.preloadImageComparisonTab()

        with gr.Blocks(title=EXT_NAME, analytics_enabled=False) as replacerUi:
            gr.Textbox(elem_id="txt2img_prompt", visible=False) # triggers onUiLoaded
            gr.Textbox(value=shared.opts.dumpjson(), elem_id="settings_json", visible=False)

            with gr.Tabs(elem_id='tabs'): # triggers progressbar
                with gr.Tab(label=f"{EXT_NAME} dedicated", elem_id=f"tab_{EXT_NAME_LOWER}_dedicated"):
                    tab = replacer_main_ui.replacerMainUI_dedicated.getReplacerTabUI()
                    tab.render()
                with gr.Tab(label="Video", elem_id=f"tab_video"):
                    tab_video = replacer_main_ui.replacerMainUI_dedicated.getReplacerVideoTabUI()
                    tab_video.render()
                if extrasInDedicated():
                    with gr.Tab(label="Extras", elem_id="extras"):
                        with gr.Blocks(analytics_enabled=False) as extras_interface:
                            ui_postprocessing.create_ui()
                replacer_extensions.image_comparison.mountImageComparisonTab()

            footer = getReplacerFooter()
            gr.HTML(footer, elem_id="footer")

        loadsave = copy.copy(demo.ui_loadsave)
        loadsave.finalized_ui = False
        video_title = f"{EXT_NAME} - video"
        loadsave.add_block(tab, EXT_NAME)
        loadsave.add_block(tab_video, video_title)
        if extrasInDedicated():
            loadsave.add_block(extras_interface, "extras")
        loadsave.dump_defaults()
        replacerUi.ui_loadsave = loadsave
        gr.mount_gradio_app(app, replacerUi, path=path)
    except Exception as e:
        errors.report(f'[{EXT_NAME}] error while creating dedicated page: {e}', exc_info=True)

script_callbacks.on_app_started(mountDedicatedPage)


def hideSegmentAnythingAccordions(component, **kwargs):
    if type(component) is gr.Accordion and\
        getattr(component, 'label', "") == "Segment Anything":

        component.visible = False
        print(f"[{EXT_NAME}] Segment Anything accordion has been hidden")

if needHideSegmentAnythingAccordions():
    script_callbacks.on_after_component(hideSegmentAnythingAccordions)


def hideAnimateDiffAccordions(component, **kwargs):
    if type(component) is gr.Accordion and\
        getattr(component, 'label', "") == "AnimateDiff":

        component.visible = False
        print(f"[{EXT_NAME}] AnimateDiff accordion has been hidden")

if needHideAnimateDiffAccordions():
    script_callbacks.on_after_component(hideAnimateDiffAccordions)


script_callbacks.on_before_ui(replacer_main_ui.initMainUI)
script_callbacks.on_after_component(replacer_extensions.controlnet.watchControlNetUI)
script_callbacks.on_after_component(replacer_extensions.soft_inpainting.watchSoftInpaintUI)
script_callbacks.on_after_component(watchOutputPanel)
script_callbacks.on_after_component(watchSetCustomScriptSourceForComponents)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(replacer_extensions.image_comparison.addButtonIntoComparisonTab)
script_callbacks.on_after_component(replacer_extensions.image_comparison.watchImageComparison)
