import gradio as gr
from replacer.ui.tools_ui import AttrDict
from replacer.tools import EXT_NAME


def getMaskComponent(num: int):
    mask = gr.Image(label="Custom mask",
            show_label=False,
            source="canvas",
            interactive=True,
            type="pil",
            tool="sketch",
            image_mode="RGBA",
            brush_color="#5f008f",
            elem_id=f'replacer_video_mask_{num}')
    return mask


def makeVideoMaskingUI(comp: AttrDict):
    with gr.Row():
        reload_masks = gr.Button("Reload masks")
        generate_masks = gr.Button("Generate new masks")
        gr.Markdown(f"All detection options including prompt are taken from {EXT_NAME} tab")
    with gr.Row(elem_id="replacer_video_masking_row_1"):
        mask1 = getMaskComponent(1)
        mask2 = getMaskComponent(2)
        mask3 = getMaskComponent(3)
        mask4 = getMaskComponent(4)
        mask5 = getMaskComponent(5)
    with gr.Row(elem_id="replacer_video_masking_row_2"):
        mask6 = getMaskComponent(6)
        mask7 = getMaskComponent(7)
        mask8 = getMaskComponent(8)
        mask9 = getMaskComponent(9)
        mask10 = getMaskComponent(10)
    with gr.Row():
        pageLabel = gr.Markdown("**Page 0/0**")
        selectedPage = gr.Number(value=0, visible=False)
        goPrev = gr.Button("← Prev. page")
        goNext = gr.Button("Next page →")
        addMasks = gr.Button("⧉ Add masks on this page")
        subMasks = gr.Button("⧉ Subtract masks on this page")
