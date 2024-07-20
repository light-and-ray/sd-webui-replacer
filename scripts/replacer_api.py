from typing import Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
import modules.script_callbacks as script_callbacks
from modules import shared
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from modules.call_queue import queue_lock
from replacer.generate import generate
from replacer.generation_args import GenerationArgs, HiresFixArgs, DUMMY_ANIMATEDIFF_ARGS
from replacer.tools import generateSeed
from replacer.ui.tools_ui import IS_WEBUI_1_9, prepareExpectedUIBehavior
from replacer.extensions import replacer_extensions




def replacer_api(_, app: FastAPI):
    from scripts.sam import sam_model_list
    from scripts.dino import dino_model_list
    try:
        from lama_cleaner_masked_content.inpaint import lamaInpaint
        lama_cleaner_available = True
    except Exception as e:
        lama_cleaner_available = False

    class ReplaceRequest(BaseModel):
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
        clip_skip: int = 1
        rotation_fix: str = '-' # choices: '-', '⟲', '⟳', '🗘'
        extra_include: list = ["mask", "box", "cut", "preview", "script"]
        variation_seed: int = -1
        variation_strength: float = 0.0
        integer_only_masked: bool = False

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
        hf_positive_prompt: str = ""
        hf_negative_prompt: str = ""
        hf_sd_model_checkpoint: str = "Use same checkpoint"
        hf_extra_inpaint_padding: int = 250
        hf_extra_mask_blur: int = 2
        hf_randomize_seed: bool = True
        hf_soft_inpaint: str = "Same"

        scripts : dict = {} # ControlNet and Soft Inpainting. See apiExample.py for example


    @app.post("/replacer/replace")
    async def api_replacer_replace(data: ReplaceRequest = Body(...)) -> Any:
        image = decode_base64_to_image(data.input_image).convert("RGBA")

        cn_args, soft_inpaint_args = replacer_extensions.prepareScriptsArgs_api(data.scripts)

        hires_fix_args = HiresFixArgs(
            upscaler = data.hf_upscaler,
            steps = data.hf_steps,
            sampler = data.hf_sampler,
            scheduler=data.hf_scheduler,
            denoise = data.hf_denoise,
            cfg_scale = data.hf_cfg_scale,
            positive_prompt_suffix = data.hf_positive_prompt_suffix,
            size_limit = data.hf_size_limit,
            above_limit_upscaler = data.hf_above_limit_upscaler,
            unload_detection_models = data.hf_unload_detection_models,
            disable_cn = data.hf_disable_cn,
            extra_mask_expand = data.hf_extra_mask_expand,
            positive_prompt = data.hf_positive_prompt,
            negative_prompt = data.hf_negative_prompt,
            sd_model_checkpoint = data.hf_sd_model_checkpoint,
            extra_inpaint_padding = data.hf_extra_inpaint_padding,
            extra_mask_blur = data.hf_extra_mask_blur,
            randomize_seed = data.hf_randomize_seed,
            soft_inpaint = data.hf_soft_inpaint,
        )

        gArgs = GenerationArgs(
            positivePrompt=data.positive_prompt,
            negativePrompt=data.negative_prompt,
            detectionPrompt=data.detection_prompt,
            avoidancePrompt=data.avoidance_prompt,
            upscalerForImg2Img=data.upscaler_for_img2img,
            seed=data.seed,
            samModel=data.sam_model_name,
            grdinoModel=data.dino_model_name,
            boxThreshold=data.box_threshold,
            maskExpand=data.mask_expand,
            maxResolutionOnDetection=data.max_resolution_on_detection,

            steps=data.steps,
            sampler_name=data.sampler,
            scheduler=data.scheduler,
            mask_blur=data.mask_blur,
            inpainting_fill=data.inpainting_fill,
            batch_count=1,
            batch_size=1,
            cfg_scale=data.cfg_scale,
            denoising_strength=data.denoise,
            height=data.height,
            width=data.width,
            inpaint_full_res_padding=data.inpaint_padding,
            img2img_fix_steps=data.fix_steps,
            inpainting_mask_invert=data.inpainting_mask_invert,

            images=[image],
            override_sd_model=True,
            sd_model_checkpoint=data.sd_model_checkpoint,
            mask_num=data.mask_num,
            avoidance_mask=None,
            only_custom_mask=False,
            custom_mask=None,
            use_inpaint_diff=False,
            clip_skip=data.clip_skip,
            pass_into_hires_fix_automatically=data.use_hires_fix,
            save_before_hires_fix=False,
            previous_frame_into_controlnet=[],
            do_not_use_mask=False,
            animatediff_args=DUMMY_ANIMATEDIFF_ARGS,
            rotation_fix=data.rotation_fix,
            variation_seed=data.variation_seed,
            variation_strength=data.variation_strength,
            integer_only_masked=data.integer_only_masked,

            hires_fix_args=hires_fix_args,
            cn_args=cn_args,
            soft_inpaint_args=soft_inpaint_args,
            )
        prepareExpectedUIBehavior(gArgs)

        with queue_lock:
            shared.state.begin('api /replacer/replace')
            try:
                processed, allExtraImages = generate(gArgs, "", False, False, data.extra_include)
            finally:
                shared.state.end()

        return {
            "image": encode_pil_to_base64(processed.images[0]).decode(),
            "extra_images": [encode_pil_to_base64(x).decode() for x in allExtraImages],
            "info": processed.info,
            "json": processed.js(),
        }


    @app.post("/replacer/available_options")
    async def api_replacer_avaliable_options() -> Any:
        return {
            "sam_model_name": sam_model_list,
            "dino_model_name": dino_model_list,
            "upscalers": [""] + [x.name for x in shared.sd_upscalers],
            "lama_cleaner_available": lama_cleaner_available, # inpainting_fill=4, https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content
            "available_scripts": replacer_extensions.getAvailableScripts_api(),
        }


script_callbacks.on_app_started(replacer_api)

