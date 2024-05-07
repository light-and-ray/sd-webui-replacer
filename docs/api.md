

### API
API is avaliable on `/replacer/replace`

```python
    input_image: str = "base64 image"
    detection_prompt: str = ""
    avoidance_prompt: str = ""
    positive_prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    sam_model_name: str = sam_model_list[0] if sam_model_list else ""
    dino_model_name: str = dino_model_list[0]
    seed: int = -1
    sampler: str = "DPM++ 2M SDE" if IS_WEBUI_1_9 else "DPM++ 2M SDE Karras"
    scheduler: str = "Automatic"
    steps: int = 20
    box_threshold: float = 0.3
    mask_expand: int = 35
    mask_blur: int = 4
    mask_num: str = "Random"
    max_resolution_on_detection = 1280
    cfg_scale: float = 5.5
    denoise: float = 1.0
    inpaint_padding = 40
    inpainting_mask_invert: bool = False
    upscaler_for_img2img : str = ""
    fix_steps : bool = False
    inpainting_fill : int = 0
    sd_model_checkpoint : str = ""
    lama_cleaner_upscaler: str = ""
    clip_skip: int = 1
    rotation_fix: str = '-' # choices: '-', '⟲', '⟳', '🗘'
    extra_include: list = ["mask", "box", "cutted", "preview", "script"]

    use_hires_fix: bool = False
    hf_upscaler: str = "ESRGAN_4x"
    hf_steps: int = 4
    hf_sampler: str = "Use same sampler"
    hf_scheduler: str = "Use same scheduler"
    hf_denoise: float = 0.35
    hf_cfg_scale: float = 1.0
    hf_positive_prompt_suffix: str = "<lora:lcm-lora-sdv1-5:1>"
    hf_size_limit: int = 1800
    hf_above_limit_upscaler: str = "Lanczos"
    hf_unload_detection_models: bool = True
    hf_disable_cn: bool = True
    hf_extra_mask_expand: int = 5
    hf_positve_prompt: str = ""
    hf_negative_prompt: str = ""
    hf_sd_model_checkpoint: str = "Use same checkpoint"
    hf_extra_inpaint_padding: int = 250
    hf_extra_mask_blur: int = 2
    hf_randomize_seed: bool = True
    hf_soft_inpaint: str = "Same"

    scripts : dict = {} # ControlNet and Soft Inpainting. See apiExample.py for example
```

Avaliable options on `/replacer/avaliable_options`

http://127.0.0.1:7860/docs#/default/api_replacer_replace_replacer_replace_post

See an example of usage in [apiExample.py](/apiExample.py) script
