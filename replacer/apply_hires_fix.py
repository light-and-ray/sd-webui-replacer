import copy
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
    hf_upscaler,
    hf_steps,
    hf_sampler,
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
    gArgs = copy.copy(generate_ui.lastGenerationArgs)
    hires_fix_args = HiresFixArgs(
        upscaler = hf_upscaler,
        steps = hf_steps,
        sampler = hf_sampler,
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
    gArgs.hires_fix_args = hires_fix_args
    gArgs.pass_into_hires_fix_automatically = False

    if generate_ui.lastGenerationArgs is None:
        return [], "", plaintext_to_html("no last generation data"), ""

    if gArgs.generationsN > 1 or gArgs.batch_size > 1 or gArgs.batch_count > 1:
        errorText = f"    [{EXT_NAME}]    applyHiresFix is not supported for batch"
        print(errorText)
        return None, "", plaintext_to_html(errorText), ""

    prepareGenerationArgsBeforeHiresFixPass(gArgs)
    hrGArgs = getGenerationArgsForHiresFixPass(gArgs)
    image = next(iter(copy.copy(gArgs.images)))

    shared.state.job_count = 2
    shared.total_tqdm.clear()
    shared.total_tqdm.updateTotal(gArgs.steps + hrGArgs.steps)

    shared.state.textinfo = "inpaint with upscaler"
    if generate_ui.lastGenerationArgs.hiresFixCacheData is not None and\
            generate_ui.lastGenerationArgs.hiresFixCacheData.upscaler == hf_upscaler:
        generatedImage = generate_ui.lastGenerationArgs.hiresFixCacheData.generatedImage
        print('hiresFixCacheData restored from cache')
        shared.state.job_count = 1
        shared.total_tqdm.updateTotal(hrGArgs.steps)
    else:
        processed, scriptImages = inpaint(image, gArgs)
        generatedImage = processed.images[0]
        if not interrupted() and not shared.state.skipped:
            generate_ui.lastGenerationArgs.hiresFixCacheData = HiresFixCacheData(hf_upscaler, generatedImage)
            print('hiresFixCacheData cached')
        

    shared.state.textinfo = "hiresfix"
    processed, scriptImages = inpaint(generatedImage, hrGArgs, getSaveDir(), "-hires-fix")

    shared.state.end()

    return processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
