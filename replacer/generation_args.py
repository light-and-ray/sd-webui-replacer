from dataclasses import dataclass
from typing import Any
from PIL import Image


@dataclass
class HiresFixArgs:
    upscaler: str
    steps: int
    sampler: str
    denoise: float
    cfg_scale: float
    positive_prompt_suffix: str
    size_limit: int
    above_limit_upscaler: str
    unload_detection_models: bool
    disable_cn: bool
    extra_mask_expand: int
    positve_prompt: str
    negative_prompt: str
    sd_model_checkpoint: str
    extra_inpaint_padding: int
    extra_mask_blur: int
    randomize_seed: bool
    soft_inpaint: str


@dataclass
class HiresFixCacheData:
    upscaler: str
    generatedImage: Image


@dataclass
class GenerationArgs:
    positvePrompt: str
    negativePrompt: str
    detectionPrompt: str
    avoidancePrompt: str
    mask: Image
    upscalerForImg2Img: str
    seed: int
    samModel: str
    grdinoModel: str
    boxThreshold: float
    maskExpand: int
    maxResolutionOnDetection: int
    steps: int
    sampler_name: str
    mask_blur: int
    inpainting_fill: int
    batch_count: int
    batch_size: int
    cfg_scale: float
    denoising_strength: float
    height: int
    width: int
    inpaint_full_res_padding: int
    img2img_fix_steps: bool
    inpainting_mask_invert : int
    images: Any
    generationsN: int
    override_sd_model: bool
    sd_model_checkpoint: str
    mask_num: int
    mask_num_for_metadata: int
    avoidance_mask: Image
    only_custom_mask: bool
    custom_mask: Image
    use_inpaint_diff: bool
    lama_cleaner_upscaler: str
    clip_skip: int

    pass_into_hires_fix_automatically: bool
    save_before_hires_fix: bool
    hires_fix_args: HiresFixArgs

    cn_args: list
    soft_inpaint_args: list

    hiresFixCacheData: HiresFixCacheData = None
    addHiresFixIntoMetadata: bool = False

