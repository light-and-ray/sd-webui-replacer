from contextlib import closing
from PIL import Image
import modules.shared as shared
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts
from modules.images import save_image
from modules import errors
from replacer.generation_args import GenerationArgs
from replacer.extensions import replacer_extensions
from replacer.tools import addReplacerMetadata, limitSizeByOneDimension, applyRotationFix, removeRotationFix
from replacer.ui.tools_ui import IS_WEBUI_1_9




def inpaint(
    image : Image.Image,
    gArgs : GenerationArgs,
    savePath : str = "",
    saveSuffix : str = "",
    save_to_dirs : bool = True,
    batch_processed : Processed = None
):
    override_settings = {}
    if gArgs.upscalerForImg2Img is not None and gArgs.upscalerForImg2Img != "" and gArgs.upscalerForImg2Img != "None":
        override_settings["upscaler_for_img2img"] = gArgs.upscalerForImg2Img
    if gArgs.sd_model_checkpoint is not None and gArgs.sd_model_checkpoint != "":
        override_settings["sd_model_checkpoint"] = gArgs.sd_model_checkpoint
    override_settings["img2img_fix_steps"] = gArgs.img2img_fix_steps
    if replacer_extensions.background_extensions.lamaCleanerAvailable():
        override_settings["upscaling_upscaler_for_lama_cleaner_masked_content"] = gArgs.lama_cleaner_upscaler
    override_settings["CLIP_stop_at_last_layers"] = gArgs.clip_skip

    mask = gArgs.mask
    if mask:
        mask = mask.resize(image.size).convert('L')
    schedulerKWargs = {"scheduler": gArgs.scheduler} if IS_WEBUI_1_9 else {}

    image = applyRotationFix(image, gArgs.rotation_fix)
    mask = applyRotationFix(mask, gArgs.rotation_fix)

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=gArgs.positivePrompt,
        negative_prompt=gArgs.negativePrompt,
        styles=[],
        sampler_name=gArgs.sampler_name,
        batch_size=gArgs.batch_size,
        n_iter=gArgs.batch_count,
        steps=gArgs.steps,
        cfg_scale=gArgs.cfg_scale,
        width=gArgs.width,
        height=gArgs.height,
        init_images=[image],
        mask=mask,
        mask_blur=gArgs.mask_blur,
        inpainting_fill=gArgs.inpainting_fill,
        resize_mode=0,
        denoising_strength=gArgs.denoising_strength,
        image_cfg_scale=1.5,
        inpaint_full_res=True,
        inpaint_full_res_padding=gArgs.inpaint_full_res_padding,
        inpainting_mask_invert=gArgs.inpainting_mask_invert,
        override_settings=override_settings,
        do_not_save_samples=True,
        **schedulerKWargs,
    )

    if gArgs.do_not_use_mask and (not gArgs.upscalerForImg2Img or gArgs.upscalerForImg2Img == "None"):
        p.inpaint_full_res = False
        p.resize_mode = 2 # resize and fill
        p.width, p.height = limitSizeByOneDimension(image.size, max(p.width, p.height))
    p.extra_generation_params["Mask blur"] = gArgs.mask_blur
    addReplacerMetadata(p, gArgs)
    p.seed = gArgs.seed
    p.do_not_save_grid = True
    try:
        if replacer_extensions.controlnet.SCRIPT and gArgs.cn_args is not None and len(gArgs.cn_args) != 0:
            previousFrame = None
            if batch_processed:
                previousFrame = batch_processed.images[-1]
            replacer_extensions.controlnet.enableInpaintModeForCN(gArgs, p, previousFrame)
    except Exception as e:
        if isinstance(e, replacer_extensions.controlnet.UnitIsReserved):
            raise
        errors.report(f"Error {e}", exc_info=True)

    replacer_extensions.applyScripts(p, gArgs)



    with closing(p):
        processed = process_images(p)

    needRestoreAfterCN = getattr(p, 'needRestoreAfterCN', False)
    if needRestoreAfterCN:
        replacer_extensions.controlnet.restoreAfterCN(image, mask, gArgs, processed)

    for i in range(len(processed.images)):
        processed.images[i] = removeRotationFix(processed.images[i], gArgs.rotation_fix)

    scriptImages = processed.images[len(processed.all_seeds):]
    processed.images = processed.images[:len(processed.all_seeds)]
    scriptImages.extend(getattr(processed, 'extra_images', []))




    if savePath:
        for i in range(len(processed.images)):
            additional_save_suffix = getattr(image, 'additional_save_suffix', None)
            suffix = saveSuffix
            if additional_save_suffix:
                suffix = additional_save_suffix + suffix
            save_image(processed.images[i], savePath, "", processed.all_seeds[i], gArgs.positivePrompt, opts.samples_format,
                    info=processed.infotext(p, i), p=p, suffix=suffix, save_to_dirs=save_to_dirs)

    if opts.do_not_show_images:
        processed.images = []

    if batch_processed:
        batch_processed.images += processed.images
        batch_processed.infotexts += processed.infotexts
        processed = batch_processed

    return processed, scriptImages

