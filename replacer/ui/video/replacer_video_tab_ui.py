import gradio as gr
from replacer.ui.tools_ui import AttrDict, OverrideCustomScriptSource

from replacer.ui.video.video_options_ui import makeVideoOptionsUI
from replacer.ui.video.video_project_ui import makeVideoProjectUI
from replacer.ui.video.video_masking_ui import makeVideoMaskingUI



def getVideoTabUI(mainTabComp: AttrDict, isDedicatedPage: bool):
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
                    makeVideoMaskingUI(comp, mainTabComp)
                with gr.Tab("Step 4 (First frame)"):
                    pass
                with gr.Tab("Step 5 (Generation)"):
                    pass
            comp.selected_project_status.render()
            comp.selected_project.render()

    return replacerVideoTabUI
