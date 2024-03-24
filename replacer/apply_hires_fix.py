import copy, json
from PIL import Image
import modules.shared as shared
from modules.ui import plaintext_to_html
from replacer.generation_args import HiresFixCacheData, HiresFixArgs
from replacer.options import EXT_NAME, getSaveDir
from replacer.tools import interrupted
from replacer import generate_ui
from replacer.inpaint import inpaint
from replacer.hires_fix import getGenerationArgsForHiresFixPass, prepareGenerationArgsBeforeHiresFixPass




def applyHiresFix(
    id_task,
    gallery_idx,
    gallery,
    generation_info,
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
    hf_positve_prompt,
    hf_negative_prompt,
    hf_sd_model_checkpoint,
    hf_extra_inpaint_padding,
    hf_extra_mask_blur,
    hf_randomize_seed,
    hf_soft_inpaint,
):
    original_gallery = []
    for i, image in enumerate(gallery):
        fake_image = Image.new(mode="RGB", size=(1, 1))
        fake_image.already_saved_as = image["name"].rsplit('?', 1)[0]
        original_gallery.append(fake_image)

    if generate_ui.lastGenerationArgs is None:
        return original_gallery, generation_info, plaintext_to_html("no last generation data"), ""

    gArgs = copy.copy(generate_ui.lastGenerationArgs)
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
        positve_prompt = hf_positve_prompt,
        negative_prompt = hf_negative_prompt,
        sd_model_checkpoint = hf_sd_model_checkpoint,
        extra_inpaint_padding = hf_extra_inpaint_padding,
        extra_mask_blur = hf_extra_mask_blur,
        randomize_seed = hf_randomize_seed,
        soft_inpaint = hf_soft_inpaint,
    )

    if len(gArgs.appropriateInputImageDataList) == 1:
        gallery_idx = 0
    if gallery_idx < 0:
        return original_gallery, generation_info, plaintext_to_html("Image for hires fix is not selected"), ""
    if gallery_idx >= len(gArgs.appropriateInputImageDataList):
        return original_gallery, generation_info, plaintext_to_html("Cannot applyhires fix for extra included images"), ""
    inputImageIdx = gArgs.appropriateInputImageDataList[gallery_idx].inputImageIdx
    image = gArgs.images[inputImageIdx]
    gArgs.mask = gArgs.appropriateInputImageDataList[gallery_idx].mask
    gArgs.seed = gArgs.appropriateInputImageDataList[gallery_idx].seed
    gArgs.hires_fix_args = hires_fix_args
    gArgs.pass_into_hires_fix_automatically = False
    gArgs.batch_count = 1
    gArgs.batch_size = 1

    prepareGenerationArgsBeforeHiresFixPass(gArgs)
    hrGArgs = getGenerationArgsForHiresFixPass(gArgs)

    shared.state.job_count = 2
    shared.total_tqdm.clear()
    shared.total_tqdm.updateTotal(gArgs.steps + hrGArgs.steps)

    shared.state.textinfo = "inpainting with upscaler"
    if generate_ui.lastGenerationArgs.hiresFixCacheData is not None and\
            generate_ui.lastGenerationArgs.hiresFixCacheData.upscaler == hf_upscaler and\
            generate_ui.lastGenerationArgs.hiresFixCacheData.galleryIdx == gallery_idx:
        generatedImage = generate_ui.lastGenerationArgs.hiresFixCacheData.generatedImage
        print('hiresFixCacheData restored from cache')
        shared.state.job_count = 1
        shared.total_tqdm.updateTotal(hrGArgs.steps)
    else:
        processed, scriptImages = inpaint(image, gArgs)
        generatedImage = processed.images[0]
        if not interrupted() and not shared.state.skipped:
            generate_ui.lastGenerationArgs.hiresFixCacheData = HiresFixCacheData(hf_upscaler, generatedImage, gallery_idx)
            print('hiresFixCacheData cached')


    shared.state.textinfo = "applying hires fix"
    processed, scriptImages = inpaint(generatedImage, hrGArgs, getSaveDir(), "-hires-fix")

    shared.state.end()

    new_gallery = []
    geninfo = json.loads(generation_info)
    for i, image in enumerate(gallery):
        if i == gallery_idx:
            geninfo["infotexts"][gallery_idx: gallery_idx+1] = processed.infotexts
            new_gallery.extend(processed.images)
        else:
            fake_image = Image.new(mode="RGB", size=(1, 1))
            fake_image.already_saved_as = image["name"].rsplit('?', 1)[0]
            new_gallery.append(fake_image)

    geninfo["infotexts"][gallery_idx] = processed.info

    return new_gallery, json.dumps(geninfo), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
