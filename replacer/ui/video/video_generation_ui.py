import gradio as gr
from replacer.ui.tools_ui import AttrDict
from replacer.tools import EXT_NAME

from .generation import videoGenerateUI

def makeVideoGenerationUI(comp: AttrDict, mainTabComp: AttrDict):
    status = gr.Markdown()
    generateButton = gr.Button("Generate")

    generateButton.click(
        fn=videoGenerateUI,
        inputs=[
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
        outputs=[status],
    )

