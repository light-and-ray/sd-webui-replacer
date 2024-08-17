import gradio as gr
from replacer.ui.tools_ui import AttrDict
from replacer.tools import EXT_NAME
from replacer.options import getVideoMaskEditingColorStr

from .masking import generateEmptyMasks, reloadMasks, goNextPage, goPrevPage, goToPage, addMasks, subMasks


def getMaskComponent(num: int):
    mask = gr.Image(label="Custom mask",
            show_label=False,
            source="upload",
            interactive=True,
            type="pil",
            tool="sketch",
            image_mode="RGB",
            brush_color=getVideoMaskEditingColorStr(),
            elem_id=f'replacer_video_mask_{num}',
            elem_classes='replacer_video_mask',
        )
    return mask


def makeVideoMaskingUI(comp: AttrDict):
    with gr.Row():
        reload_masks = gr.Button("⟳ Reload page")
        generate_empty_masks = gr.Button("Generate empty masks")
        generate_detected_masks = gr.Button("Generate detected masks")
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
        pageLabel = gr.Markdown("**Page 0/0**", elem_id="replacer_video_masking_page_label")
        selectedPage = gr.Number(value=0, visible=False, precision=0)
        goPrev = gr.Button("← Prev. page")
        goNext = gr.Button("Next page →")
        addMasksButton = gr.Button("⧉ Add masks on this page")
        subMasksButton = gr.Button("⧉ Subtract masks on this page")
    with gr.Row():
        pageToGo = gr.Number(label="Page to go", value=1, precision=0, minimum=1)
        goToPageButton = gr.Button("Go to page")
        gr.Markdown('Quality of masks preview is reduced for performance\nIf you see broken images, just click "Reload page"')


    generate_empty_masks.click(
        fn=generateEmptyMasks,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, comp.target_video_fps, comp.ad_generate_only_first_fragment, comp.ad_fragment_length],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
        )
    reload_masks.click(
        fn=reloadMasks,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
        )
    goPrev.click(
        fn=goPrevPage,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
        )
    goNext.click(
        fn=goNextPage,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
        )
    goToPageButton.click(
        fn=goToPage,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, pageToGo],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
        )



    addMasksButton.click(
        fn=addMasks,
        inputs=[comp.selected_project, selectedPage, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
    ).then(
        fn=reloadMasks,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
    )

    subMasksButton.click(
        fn=subMasks,
        inputs=[comp.selected_project, selectedPage, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
    ).then(
        fn=reloadMasks,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
    )

