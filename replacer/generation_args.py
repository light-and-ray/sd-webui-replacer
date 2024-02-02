from dataclasses import dataclass
from typing import Any

@dataclass
class GenerationArgs:
    positvePrompt: Any
    negativePrompt: Any
    detectionPrompt: Any
    avoidancePrompt: Any
    mask: Any
    upscalerForImg2Img: Any
    seed: Any
    samModel: Any
    grdinoModel: Any
    boxThreshold: Any
    maskExpand: Any
    maxResolutionOnDetection: Any
    steps: Any
    sampler_name: Any
    mask_blur: Any
    inpainting_fill: Any
    n_iter: Any
    batch_size: Any
    cfg_scale: Any
    denoising_strength: Any
    height: Any
    width: Any
    inpaint_full_res_padding: Any
    img2img_fix_steps: Any
    inpainting_mask_invert : Any
    images: Any
    generationsN: Any
    sd_model_checkpoint : Any
    mask_num : Any
    mask_num_for_metadata : Any
    avoidance_mask : Any
    only_custom_mask : Any
    custom_mask : Any
    cn_args : Any
