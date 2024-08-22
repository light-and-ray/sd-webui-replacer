import gradio as gr
from replacer.ui.tools_ui import AttrDict, OverrideCustomScriptSource
from replacer.tools import Pause

from replacer.ui.video.video_options_ui import makeVideoOptionsUI
from replacer.ui.video.video_project_ui import makeVideoProjectUI
from replacer.ui.video.video_masking_ui import makeVideoMaskingUI
from replacer.ui.video.video_generation_ui import makeVideoGenerationUI


def getVideoTabUI(mainTabComp: AttrDict, isDedicatedPage: bool):
    comp = AttrDict()
    with OverrideCustomScriptSource('Video'):
        comp.selected_project_status = gr.Markdown("❌ Project is not selected", elem_id="replacer_video_selected_project_status")
        comp.selected_project = gr.Textbox(visible=False)

        with gr.Blocks(analytics_enabled=False) as replacerVideoTabUI:
            with gr.Tabs():
                with gr.Tab("Step 1 (Project)"):
                    makeVideoProjectUI(comp)
                with gr.Tab("Step 2 (Options)"):
                    makeVideoOptionsUI(comp)
                with gr.Tab("Step 3 (Masking)", elem_id="replacer_video_masking_tab"):
                    makeVideoMaskingUI(comp, mainTabComp)
                with gr.Tab("Step 4 (Generation)"):
                    makeVideoGenerationUI(comp, mainTabComp)
            with gr.Row():
                comp.selected_project_status.render()
                comp.selected_project.render()
                comp.pause_button = gr.Button(
                f'pause/resume video generation',
                    elem_id='replacer_video_pause',
                    visible=True,
                    elem_classes=["replacer-pause-button"],
                    variant='compact'
                )
                comp.pause_button.click(
                    fn=Pause.toggle
                )

    return replacerVideoTabUI
