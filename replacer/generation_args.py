from dataclasses import dataclass
from typing import Any
from PIL import Image


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
    n_iter: int
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
    sd_model_checkpoint: str
    mask_num: int
    mask_num_for_metadata: int
    avoidance_mask: Image
    only_custom_mask: bool
    custom_mask: Image
    use_inpaint_diff: bool
    cn_args: list
    soft_inpaint_args: list

    hiresFixCacheData : HiresFixCacheData = None
