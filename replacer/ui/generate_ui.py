import os, datetime, copy
from PIL import Image
import modules.shared as shared
from modules.ui import plaintext_to_html
from replacer.generation_args import GenerationArgs, HiresFixArgs, HiresFixCacheData, AnimateDiffArgs
from replacer.video_tools import getVideoFrames, save_video, overrideSettingsForVideo
from replacer.options import getSaveDir
from replacer.extensions import replacer_extensions
from replacer.tools import prepareMask, generateSeed, convertIntoPath
from replacer.ui.tools_ui import prepareExpectedUIBehavior
from replacer.generate import generate
from replacer.video_animatediff import animatediffGenerate


lastGenerationArgs: GenerationArgs = None

def getLastUsedSeed():
    if lastGenerationArgs is None:
        return -1
    else:
        return lastGenerationArgs.seed


def getLastUsedVariationSeed():
    if lastGenerationArgs is None:
        return -1
    else:
        return lastGenerationArgs.variation_seed


def getLastUsedMaskNum():
    if lastGenerationArgs is None or not lastGenerationArgs.mask_num_for_metadata:
        return "Random"
    else:
        return str(lastGenerationArgs.mask_num_for_metadata)


def generate_ui_(
    id_task,
    selected_input_mode: str,
    detectionPrompt: str,
    avoidancePrompt: str,
    positivePrompt: str,
    negativePrompt: str,
    image_single,
    image_batch,
    keep_original_filenames,
    input_batch_dir: str,
    output_batch_dir: str,
    keep_original_filenames_from_dir,
    show_batch_dir_results,
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
    batch_count,
    batch_size,
    inpainting_mask_invert,
    extra_includes,
    fix_steps,
    override_sd_model,
    sd_model_checkpoint,
    mask_num,
    avoidance_mask_mode,
    avoidance_mask,
    only_custom_mask,
    custom_mask_mode,
    custom_mask,
    use_inpaint_diff,
    inpaint_diff_mask_view,
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

    hf_upscaler,
    hf_steps,
    hf_sampler,
    hf_scheduler,
    hf_denoise,
    hf_cfg_scale,
    hf_positive_prompt_suffix,
    hf_size_limit,
    hf_above_limit_upscaler,
    hf_unload_detection_models,
    hf_disable_cn,
    hf_extra_mask_expand,
    hf_positive_prompt,
    hf_negative_prompt,
    hf_sd_model_checkpoint,
    hf_extra_inpaint_padding,
    hf_extra_mask_blur,
    hf_randomize_seed,
    hf_soft_inpaint,
    hf_supersampling,

    *scripts_args,
):
    if (seed == -1):
        seed = generateSeed()

    input_batch_dir = convertIntoPath(input_batch_dir)
    output_batch_dir = convertIntoPath(output_batch_dir)

    images = []

    if selected_input_mode == "tab_single":
        if image_single is not None:
            images = [image_single]

    if selected_input_mode == "tab_batch":
        def getImages(image_folder):
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = img
                else:
                    filename = os.path.abspath(img.name)
                    image = Image.open(filename).convert('RGBA')
                    if keep_original_filenames:
                        image.additional_save_suffix = '-' + os.path.basename(filename)
                yield image
        if image_batch is not None:
            images = getImages(image_batch)

    if selected_input_mode == "tab_batch_dir":
        assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
        def readImages(input_dir):
            if hasattr(shared, 'walk_image_files'): # webui 1.10
                image_list = shared.walk_image_files(input_dir)
            else:
                image_list = shared.listfiles(input_dir)

            for filename in image_list:
                try:
                    image = Image.open(filename).convert('RGBA')
                    if keep_original_filenames_from_dir:
                        image.additional_save_suffix = '-' + os.path.basename(filename)
                except Exception:
                    continue
                yield image
        images = readImages(input_batch_dir)

    # timestamp = int(datetime.datetime.now().timestamp())
    # if selected_input_mode == "tab_batch_video":
    #     assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
    #     shared.state.textinfo = 'video preparing'
    #     if video_output_dir == "":
    #         video_output_dir = os.path.join(os.path.dirname(input_video), f'out_{seed}_{timestamp}')
    #     else:
    #         video_output_dir = os.path.join(video_output_dir, f'out_{seed}_{timestamp}')
    #     if os.path.exists(video_output_dir):
    #         for file in os.listdir(video_output_dir):
    #             if file.endswith(f'.{shared.opts.samples_format}'):
    #                 os.remove(os.path.join(video_output_dir, file))
    #     batch_count = 1
    #     batch_size = 1
    #     extra_includes = []
    #     save_before_hires_fix = False
    #     hf_randomize_seed = False

    #     images, fps_in, fps_out = getVideoFrames(input_video, target_video_fps)
    #     resultFrames = os.path.join(video_output_dir, 'resultFrames')


    images = list(images)

    if len(images) == 0:
        return [], "", plaintext_to_html("no input images"), ""

    cn_args, soft_inpaint_args = replacer_extensions.prepareScriptsArgs(scripts_args)

    hires_fix_args = HiresFixArgs(
        upscaler = hf_upscaler,
        steps = hf_steps,
        sampler = hf_sampler,
        scheduler = hf_scheduler,
        denoise = hf_denoise,
        cfg_scale = hf_cfg_scale,
        positive_prompt_suffix = hf_positive_prompt_suffix,
        size_limit = hf_size_limit,
        above_limit_upscaler = hf_above_limit_upscaler,
        unload_detection_models = hf_unload_detection_models,
        disable_cn = hf_disable_cn,
        extra_mask_expand = hf_extra_mask_expand,
        positive_prompt = hf_positive_prompt,
        negative_prompt = hf_negative_prompt,
        sd_model_checkpoint = hf_sd_model_checkpoint,
        extra_inpaint_padding = hf_extra_inpaint_padding,
        extra_mask_blur = hf_extra_mask_blur,
        randomize_seed = hf_randomize_seed,
        soft_inpaint = hf_soft_inpaint,
        supersampling = hf_supersampling,
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
        batch_count=batch_count,
        batch_size=batch_size,
        cfg_scale=cfg_scale,
        denoising_strength=denoise,
        height=height,
        width=width,
        inpaint_full_res_padding=inpaint_padding,
        img2img_fix_steps=fix_steps,
        inpainting_mask_invert=inpainting_mask_invert,

        images=images,
        override_sd_model=override_sd_model,
        sd_model_checkpoint=sd_model_checkpoint,
        mask_num=mask_num,
        avoidance_mask=prepareMask(avoidance_mask_mode, avoidance_mask),
        only_custom_mask=only_custom_mask,
        custom_mask=prepareMask(custom_mask_mode, custom_mask),
        use_inpaint_diff=use_inpaint_diff and inpaint_diff_mask_view is not None and \
            replacer_extensions.inpaint_difference.Globals is not None and \
            replacer_extensions.inpaint_difference.Globals.generated_mask is not None,
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

        hires_fix_args=hires_fix_args,
        cn_args=cn_args,
        soft_inpaint_args=soft_inpaint_args,
        )
    prepareExpectedUIBehavior(gArgs)



    saveDir = getSaveDir()
    saveToSubdirs = shared.opts.save_to_dirs
    if selected_input_mode == "tab_batch_dir" and output_batch_dir != "":
        saveDir = output_batch_dir
        saveToSubdirs = False


    processed, allExtraImages = generate(gArgs, saveDir, saveToSubdirs, extra_includes)
    if processed is None or not getattr(processed, 'images', None):
        return [], "", plaintext_to_html(f"No one image was processed. See console logs for exceptions"), ""

    global lastGenerationArgs
    gArgs.appropriateInputImageDataList = [x.appropriateInputImageData for x in processed.images]
    lastGenerationArgs = gArgs
    lastGenerationArgs.hiresFixCacheData = HiresFixCacheData(gArgs.upscalerForImg2Img, processed.images[0], 0)



    # if selected_input_mode == "tab_batch_video":
    #     shared.state.textinfo = 'video saving'
    #     print("generate done, generating video")
    #     save_video_path = os.path.join(video_output_dir, f'output_{os.path.basename(input_video)}_{seed}_{timestamp}.mp4')
    #     if len(save_video_path) > 260:
    #         save_video_path = os.path.join(video_output_dir, f'output_{seed}_{timestamp}.mp4')
    #     save_video(resultFrames, fps_out, input_video, save_video_path, seed)
    #     return [], "", plaintext_to_html(f"Saved as {save_video_path}"), ""

    if selected_input_mode == "tab_batch_dir" and not show_batch_dir_results:
        return [], "", plaintext_to_html(f"Saved into {saveDir}"), ""

    processed.images += allExtraImages
    processed.infotexts += [processed.info] * len(allExtraImages)

    return processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def generate_ui(*args, **kwargs):
    return generate_ui_(*args, **kwargs)



