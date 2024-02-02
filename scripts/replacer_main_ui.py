import copy
import gradio as gr
from modules import script_callbacks, progress, shared
from replacer.options import (EXT_NAME, EXT_NAME_LOWER, needHideSegmentAnythingAccordions,
)
from replacer.tab_ui import getReplacerTabUI



def on_ui_tabs():
    replacerTabUi = getReplacerTabUI(isDedicatedPage=False)
    return [(replacerTabUi, EXT_NAME, EXT_NAME)]

script_callbacks.on_ui_tabs(on_ui_tabs)


def mountDedicatedPage(demo, app):
    try:
        path = f'/{EXT_NAME_LOWER}-dedicated'
        app.add_api_route(f"{path}/internal/progress",
            progress.progressapi, methods=["POST"],
            response_model=progress.ProgressResponse)

        with gr.Blocks(title=EXT_NAME) as replacerUi:
            gr.Textbox(elem_id="txt2img_prompt", visible=False) # triggers onUiLoaded
            gr.Textbox(value=shared.opts.dumpjson(), elem_id="settings_json", visible=False)

            with gr.Tabs(elem_id='tabs'): # triggers progressbar
                with gr.Tab(label=f"{EXT_NAME} dedicated", elem_id=f"tab_{EXT_NAME}_dedicated"):
                    getReplacerTabUI(isDedicatedPage=True)
        
        loadsave = copy.copy(demo.ui_loadsave)  
        loadsave.finalized_ui = False
        loadsave.add_block(replacerUi, EXT_NAME)
        loadsave.dump_defaults()
        replacerUi.ui_loadsave = loadsave
        gr.mount_gradio_app(app, replacerUi, path=path)
    except Exception as e:
        print(f'[{EXT_NAME}] error while creating dedicated page: {e}')

script_callbacks.on_app_started(mountDedicatedPage)


def hideSegmentAnythingAccordions(demo, app):
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


if needHideSegmentAnythingAccordions():
    script_callbacks.on_app_started(hideSegmentAnythingAccordions)


