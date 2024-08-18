import datetime, os

from modules import shared
from modules.ui import plaintext_to_html
import gradio as gr
from replacer.generation_args import GenerationArgs, DUMMY_HIRESFIX_ARGS, AnimateDiffArgs
from replacer.ui.tools_ui import prepareExpectedUIBehavior
from replacer.extensions import replacer_extensions
from replacer.video_tools import overrideSettingsForVideo, save_video
from replacer.options import EXT_NAME_LOWER

from .project import getFrames, getMasks, getOriginalVideoPath
from replacer.video_animatediff import animatediffGenerate


def videoGenerateUI(
    task_id: str,
    project_path: str,
    target_video_fps: int,

    ad_fragment_length,
    ad_internal_fps,
    ad_batch_size,
    ad_stride,
    ad_overlap,
    ad_latent_power,
    ad_latent_scale,
    ad_generate_only_first_fragment,
    ad_cn_inpainting_model,
    ad_control_weight,
    ad_force_override_sd_model,
    ad_force_sd_model_checkpoint,
    ad_motion_model,

    detectionPrompt: str,
    avoidancePrompt: str,
    positivePrompt: str,
    negativePrompt: str,
    upscalerForImg2Img,
    seed,
    sampler,
    scheduler,
    steps,
    box_threshold,
    mask_expand,
    mask_blur,
    max_resolution_on_detection,
    sam_model_name,
    dino_model_name,
    cfg_scale,
    denoise,
    inpaint_padding,
    inpainting_fill,
    width,
    height,
    inpainting_mask_invert,
    fix_steps,
    override_sd_model,
    sd_model_checkpoint,
    mask_num,
    only_custom_mask,
    clip_skip,
    pass_into_hires_fix_automatically,
    save_before_hires_fix,
    do_not_use_mask,
    rotation_fix: str,
    variation_seed: int,
    variation_strength: float,
    integer_only_masked: bool,
    forbid_too_small_crop_region: bool,
    correct_aspect_ratio: bool,

    *scripts_args,
):
    cn_args, soft_inpaint_args = replacer_extensions.prepareScriptsArgs(scripts_args)

    animatediff_args = AnimateDiffArgs(
        fragment_length=ad_fragment_length,
        internal_fps=ad_internal_fps,
        batch_size=ad_batch_size,
        stride=ad_stride,
        overlap=ad_overlap,
        latent_power=ad_latent_power,
        latent_scale=ad_latent_scale,
        generate_only_first_fragment=ad_generate_only_first_fragment,
        cn_inpainting_model=ad_cn_inpainting_model,
        control_weight=ad_control_weight,
        force_override_sd_model=ad_force_override_sd_model,
        force_sd_model_checkpoint=ad_force_sd_model_checkpoint,
        motion_model=ad_motion_model,
    )

    gArgs = GenerationArgs(
        positivePrompt=positivePrompt,
        negativePrompt=negativePrompt,
        detectionPrompt=detectionPrompt,
        avoidancePrompt=avoidancePrompt,
        upscalerForImg2Img=upscalerForImg2Img,
        seed=seed,
        samModel=sam_model_name,
        grdinoModel=dino_model_name,
        boxThreshold=box_threshold,
        maskExpand=mask_expand,
        maxResolutionOnDetection=max_resolution_on_detection,

        steps=steps,
        sampler_name=sampler,
        scheduler=scheduler,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        batch_count=1,
        batch_size=1,
        cfg_scale=cfg_scale,
        denoising_strength=denoise,
        height=height,
        width=width,
        inpaint_full_res_padding=inpaint_padding,
        img2img_fix_steps=fix_steps,
        inpainting_mask_invert=inpainting_mask_invert,

        images=[],
        override_sd_model=override_sd_model,
        sd_model_checkpoint=sd_model_checkpoint,
        mask_num=mask_num,
        avoidance_mask=None,
        only_custom_mask=only_custom_mask,
        custom_mask=None,
        use_inpaint_diff=False,
        clip_skip=clip_skip,
        pass_into_hires_fix_automatically=pass_into_hires_fix_automatically,
        save_before_hires_fix=save_before_hires_fix,
        do_not_use_mask=do_not_use_mask,
        rotation_fix=rotation_fix,
        variation_seed=variation_seed,
        variation_strength=variation_strength,
        integer_only_masked=integer_only_masked,
        forbid_too_small_crop_region=forbid_too_small_crop_region,
        correct_aspect_ratio=correct_aspect_ratio,

        hires_fix_args=DUMMY_HIRESFIX_ARGS,
        cn_args=cn_args,
        soft_inpaint_args=soft_inpaint_args,

        animatediff_args=animatediff_args,
        )
    prepareExpectedUIBehavior(gArgs)

    originalVideo = getOriginalVideoPath(project_path)
    if not originalVideo:
        raise gr.Error("This project doesn't have original video")
    timestamp = int(datetime.datetime.now().timestamp())
    fragmentsPath = os.path.join(project_path, 'outputs', str(timestamp))
    resultPath = os.path.join(fragmentsPath, "result")
    frames = getFrames(project_path)
    masks = getMasks(project_path)
    if not frames or not masks:
        raise gr.Error("This project doesn't have frames or masks")
    frames = list(frames)
    masks = list(masks)
    saveVideoPath = os.path.join(fragmentsPath, f'{EXT_NAME_LOWER}_{os.path.basename(originalVideo)}_{timestamp}.mp4')
    if len(saveVideoPath) > 260:
        saveVideoPath = os.path.join(fragmentsPath, f'{EXT_NAME_LOWER}_{timestamp}.mp4')

    restore = overrideSettingsForVideo()
    try:
        animatediffGenerate(gArgs, fragmentsPath, resultPath, frames, masks, target_video_fps)

        shared.state.textinfo = 'video saving'
        print("video saving")
        save_video(resultPath, target_video_fps, originalVideo, saveVideoPath, gArgs.seed)
    finally:
        restore()

    return [], "", plaintext_to_html(f"Saved as {saveVideoPath}"), ""

