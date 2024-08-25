import gradio as gr
from replacer.ui.tools_ui import AttrDict, getSubmitJsFunction, ui_toprow, IS_WEBUI_1_8
from replacer.tools import EXT_NAME
from replacer.options import getVideoMaskEditingColorStr
from modules.ui_common import create_output_panel, update_generation_info
from modules.call_queue import wrap_gradio_gpu_call

from .masking import generateEmptyMasks, reloadMasks, goNextPage, goPrevPage, goToPage, addMasks, subMasks, generateDetectedMasks


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


def makeVideoMaskingUI(comp: AttrDict, mainTabComp: AttrDict):
    with gr.Row():
        reload_masks = gr.Button("⟳ Reload page")
        if ui_toprow:
            toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part='replacer_video_masks_detect')
            toprow.create_inline_toprow_image()
            generate_detected_masks = toprow.submit
            generate_detected_masks.variant = 'secondary'
            generate_detected_masks.value = 'Generate detected masks'
        else:
            generate_detected_masks = gr.Button('Generate detected masks', elem_id='replacer_video_masks_detect_generate')
        gr.Markdown(f"All detection options including prompt are taken from {EXT_NAME} tab. "
                    "You can stop masking at any time, and it will cut output video")
        generate_empty_masks = gr.Button("Generate empty masks")
    with gr.Row():
        if IS_WEBUI_1_8:
            outputPanel = create_output_panel('replacer_video_masking_progress', "")
            replacer_gallery = outputPanel.gallery
            generation_info = outputPanel.generation_info
            html_info = outputPanel.infotext
            html_log = outputPanel.html_log
        else:
            replacer_gallery, generation_info, html_info, html_log = \
                create_output_panel('replacer_video_masking_progress', "")
        generation_info_button = gr.Button(visible=False, elem_id="replacer_video_masking_progress_info_button")
        generation_info_button.click(
            fn=update_generation_info,
            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
            inputs=[generation_info, html_info, html_info],
            outputs=[html_info, html_info],
            show_progress=False,
        )
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
        gr.Markdown('Quality of masks preview is reduced for performance\n'
                    'If you see broken images, just click "Reload page"')
        gr.Markdown('You can copy old masks from project nested directory\n'
                    'Fps affects on masks')



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
        inputs=[comp.selected_project, selectedPage, mainTabComp.mask_blur, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
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
        inputs=[comp.selected_project, selectedPage, mainTabComp.mask_blur, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10,],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
    ).then(
        fn=reloadMasks,
        _js='closeAllVideoMasks',
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
    )





    generate_empty_masks.click(
        fn=lambda: None,
        _js='closeAllVideoMasks',
    ).then(
        fn=generateEmptyMasks,
        inputs=[mainTabComp.dummy_component, comp.selected_project, comp.target_video_fps, comp.ad_generate_only_first_fragment, comp.ad_fragment_length],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
        )


    generate_detected_masks.click(
        fn=lambda: None,
        _js='closeAllVideoMasks',
    ).then(
        fn=wrap_gradio_gpu_call(generateDetectedMasks, extra_outputs=[None, '', '']),
        _js=getSubmitJsFunction('replacer_video_masking_progress', 'replacer_video_masks_detect', '', False),
        inputs=[mainTabComp.dummy_component, comp.selected_project, comp.target_video_fps, comp.ad_generate_only_first_fragment, comp.ad_fragment_length,
            mainTabComp.detectionPrompt,
            mainTabComp.avoidancePrompt,
            mainTabComp.seed,
            mainTabComp.sam_model_name,
            mainTabComp.dino_model_name,
            mainTabComp.box_threshold,
            mainTabComp.mask_expand,
            mainTabComp.mask_blur,
            mainTabComp.max_resolution_on_detection,
            mainTabComp.inpainting_mask_invert,
            mainTabComp.mask_num,
            mainTabComp.avoid_mask_mode,
            mainTabComp.avoidance_mask,
            mainTabComp.only_custom_mask,
            mainTabComp.custom_mask_mode,
            mainTabComp.custom_mask,
            mainTabComp.do_not_use_mask,
        ],
        outputs=[
            replacer_gallery,
            generation_info,
            html_info,
            html_log,
        ],
        show_progress=ui_toprow is None,
    ).then(
        fn=reloadMasks,
        inputs=[comp.selected_project, selectedPage],
        outputs=[selectedPage, pageLabel, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10],
        postprocess=False,
    )

