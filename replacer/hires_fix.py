import copy
from replacer.generation_args import GenerationArgs
from replacer.options import getHiresFixPositivePromptSuffixExamples
from replacer.tools import clearCache, generateSeed, extraMaskExpand



def prepareGenerationArgsBeforeHiresFixPass(gArgs: GenerationArgs) -> None:
    hf = gArgs.hires_fix_args
    gArgs.upscalerForImg2Img = hf.upscaler


def getGenerationArgsForHiresFixPass(gArgs: GenerationArgs) -> GenerationArgs:
    image = next(iter(copy.copy(gArgs.images)))
    hf = gArgs.hires_fix_args
    if hf.positive_prompt_suffix == "":
        hf.positive_prompt_suffix = getHiresFixPositivePromptSuffixExamples()[0]
    hrGArgs = copy.copy(gArgs)
    hrGArgs.upscalerForImg2Img = hf.above_limit_upscaler
    hrGArgs.cfg_scale = hf.cfg_scale
    hrGArgs.denoising_strength = hf.denoise
    if not hf.sampler == 'Use same sampler':
        hrGArgs.sampler_name = hf.sampler
    if not hf.scheduler == 'Use same scheduler':
        hrGArgs.scheduler == hf.scheduler
    if hf.steps != 0:
        hrGArgs.steps = hf.steps
    if hf.extra_mask_expand != 0:
        hrGArgs.mask = extraMaskExpand(hrGArgs.mask, hf.extra_mask_expand)
    hrGArgs.inpainting_fill = 1 # Original
    hrGArgs.img2img_fix_steps = True
    if hf.disable_cn:
        hrGArgs.cn_args = None
    if hf.soft_inpaint != 'Same' and hrGArgs.soft_inpaint_args is not None and len(hrGArgs.soft_inpaint_args) != 0:
        hrGArgs.soft_inpaint_args = list(hrGArgs.soft_inpaint_args)
        hrGArgs.soft_inpaint_args[0] = hf.soft_inpaint == 'Enable'
    if hf.positve_prompt != "":
        hrGArgs.positvePrompt = hf.positve_prompt
    hrGArgs.positvePrompt = hrGArgs.positvePrompt + " " + hf.positive_prompt_suffix
    if hf.negative_prompt != "":
        hrGArgs.negativePrompt = hf.negative_prompt
    if hf.sd_model_checkpoint is not None and hf.sd_model_checkpoint != 'Use same model'\
            and hf.sd_model_checkpoint != 'Use same checkpoint' and hf.sd_model_checkpoint != "":
        hrGArgs.sd_model_checkpoint = hf.sd_model_checkpoint
    hrGArgs.inpaint_full_res_padding += hf.extra_inpaint_padding
    hrGArgs.mask_blur += hf.extra_mask_blur
    if hf.randomize_seed:
        hrGArgs.seed = generateSeed()

    if hf.unload_detection_models:
        clearCache()
    
    hrGArgs.width, hrGArgs.height = image.size
    if hrGArgs.height > hf.size_limit:
        hrGArgs.height = hf.size_limit
    if hrGArgs.width > hf.size_limit:
        hrGArgs.width = hf.size_limit
    
    hrGArgs.batch_count = 1
    hrGArgs.batch_size = 1
    hrGArgs.addHiresFixIntoMetadata = True

    return hrGArgs
