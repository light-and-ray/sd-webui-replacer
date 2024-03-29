import importlib
import gradio as gr
from replacer.options import EXT_NAME



# --- ImageComparison ---- https://github.com/Haoming02/sd-webui-image-comparison


def addButtonIntoComparisonTab(component, **kwargs):
    elem_id = kwargs.get('elem_id', None)
    if elem_id == 'img_comp_extras':
        column = component.parent
        with column.parent:
            with column:
                replacer_btn = gr.Button(f'Compare {EXT_NAME}', elem_id='img_comp_replacer')
        replacer_btn.click(None, None, None, _js='replacer_imageComparisonloadImage')


needWatchImageComparison = False

def watchImageComparison(component, **kwargs):
    global needWatchImageComparison
    if not needWatchImageComparison:
        return
    elem_id = kwargs.get('elem_id', None)
    if elem_id in ['img_comp_i2i', 'img_comp_inpaint', 'img_comp_extras']:
        component.visible = False


ImageComparisonTab = None

def preloadImageComparisonTab():
    global ImageComparisonTab, needWatchImageComparison
    try:
        img_comp = importlib.import_module('extensions.sd-webui-image-comparison.scripts.img_comp')
    except ImportError:
        return
    needWatchImageComparison = True
    ImageComparisonTab = img_comp.img_ui()[0]

def mountImageComparisonTab():
    global ImageComparisonTab, needWatchImageComparison
    if not ImageComparisonTab:
        return
    gr.Radio(value="Off", elem_id="setting_comp_send_btn", choices=["Off", "Text", "Icon"], visible=False)
    gr.Textbox(elem_id="replacer_image_comparison", visible=False)
    interface, label, ifid = ImageComparisonTab
    with gr.Tab(label=label, elem_id=f"tab_{ifid}"):
        interface.render()
    needWatchImageComparison = False

