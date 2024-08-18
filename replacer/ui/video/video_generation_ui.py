import gradio as gr
from replacer.ui.tools_ui import AttrDict, getSubmitJsFunction, ui_toprow, IS_WEBUI_1_8
from modules.ui_common import create_output_panel, update_generation_info
from modules.call_queue import wrap_gradio_gpu_call

from .generation import videoGenerateUI

def makeVideoGenerationUI(comp: AttrDict, mainTabComp: AttrDict):
    with gr.Row():
        if IS_WEBUI_1_8:
            outputPanel = create_output_panel('replacer_video', "")
            comp.replacer_gallery = outputPanel.gallery
            comp.generation_info = outputPanel.generation_info
            comp.html_info = outputPanel.infotext
            comp.html_log = outputPanel.html_log
        else:
            comp.replacer_gallery, comp.generation_info, comp.html_info, comp.html_log = \
                create_output_panel('replacer_video', "")
        comp.generation_info_button = gr.Button(visible=False, elem_id="replacer_video_generation_info_button")
        comp.generation_info_button.click(
            fn=update_generation_info,
            _js="function(x, y, z){ return [x, y, selected_gallery_index()] }",
            inputs=[comp.generation_info, comp.html_info, comp.html_info],
            outputs=[comp.html_info, comp.html_info],
            show_progress=False,
        )

    with gr.Row():
        if ui_toprow:
            toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part='replacer_video_gen')
            toprow.create_inline_toprow_image()
            generateButton = toprow.submit
            generateButton.variant = 'secondary'
            generateButton.value = 'Generate 🎬'
        else:
            generateButton = gr.Button('Generate 🎬', elem_id='replacer_video_gen_generate')

    generateButton.click(
        _js=getSubmitJsFunction('replacer_video', 'replacer_video_gen', '', False),
        fn=wrap_gradio_gpu_call(videoGenerateUI, extra_outputs=[None, '', '']),
        inputs=[
            mainTabComp.dummy_component, # task_id
            comp.selected_project,
            comp.target_video_fps,

            comp.ad_fragment_length,
            comp.ad_internal_fps,
            comp.ad_batch_size,
            comp.ad_stride,
            comp.ad_overlap,
            comp.ad_latent_power,
            comp.ad_latent_scale,
            comp.ad_generate_only_first_fragment,
            comp.ad_cn_inpainting_model,
            comp.ad_control_weight,
            comp.ad_force_override_sd_model,
            comp.ad_force_sd_model_checkpoint,
            comp.ad_motion_model,

            mainTabComp.detectionPrompt,
            mainTabComp.avoidancePrompt,
            mainTabComp.positivePrompt,
            mainTabComp.negativePrompt,
            mainTabComp.upscaler_for_img2img,
            mainTabComp.seed,
            mainTabComp.sampler,
            mainTabComp.scheduler,
            mainTabComp.steps,
            mainTabComp.box_threshold,
            mainTabComp.mask_expand,
            mainTabComp.mask_blur,
            mainTabComp.max_resolution_on_detection,
            mainTabComp.sam_model_name,
            mainTabComp.dino_model_name,
            mainTabComp.cfg_scale,
            mainTabComp.denoise,
            mainTabComp.inpaint_padding,
            mainTabComp.inpainting_fill,
            mainTabComp.width,
            mainTabComp.height,
            mainTabComp.inpainting_mask_invert,
            mainTabComp.fix_steps,
            mainTabComp.override_sd_model,
            mainTabComp.sd_model_checkpoint,
            mainTabComp.mask_num,
            mainTabComp.only_custom_mask,
            mainTabComp.clip_skip,
            mainTabComp.pass_into_hires_fix_automatically,
            mainTabComp.save_before_hires_fix,
            mainTabComp.do_not_use_mask,
            mainTabComp.rotation_fix,
            mainTabComp.variation_seed,
            mainTabComp.variation_strength,
            mainTabComp.integer_only_masked,
            mainTabComp.forbid_too_small_crop_region,
            mainTabComp.correct_aspect_ratio,
        ] + mainTabComp.cn_inputs
          + mainTabComp.soft_inpaint_inputs,
        outputs=[
            comp.replacer_gallery,
            comp.generation_info,
            comp.html_info,
            comp.html_log,
        ],
        show_progress=ui_toprow is None,
    )

