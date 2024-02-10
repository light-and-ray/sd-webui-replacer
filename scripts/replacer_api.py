from typing import Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
import modules.script_callbacks as script_callbacks
from modules import shared
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from replacer.generate import generate




def replacer_api(_, app: FastAPI):
    from scripts.sam import sam_model_list
    from scripts.dino import dino_model_list
    try:
        from lama_cleaner_masked_content.inpaint import lamaInpaint
        lama_cleaner_avaliable = True
    except Exception as e:
        lama_cleaner_avaliable = False

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
        sampler: str = "DPM++ 2M SDE Karras"
        steps: int = 20
        box_threshold: float = 0.3
        mask_expand: int = 35
        mask_blur: int = 4
        max_resolution_on_detection = 1280
        cfg_scale: float = 5.5
        denoise: int = 1
        inpaint_padding = 40
        inpainting_mask_invert: bool = False
        upscaler_for_img2img : str = ""
        fix_steps : bool = False
        inpainting_fill : int = 0
        sd_model_checkpoint : str = ""


    @app.post("/replacer/replace")
    async def api_replacer_replace(data: ReplaceRequest = Body(...)) -> Any:
        image = decode_base64_to_image(data.input_image).convert("RGBA")
        
        result = generate(
            data.detection_prompt, data.avoidance_prompt, data.positive_prompt, data.negative_prompt,
            0, image, [], False, "", "", False, False, "", "", 0, data.upscaler_for_img2img,
            data.seed, data.sampler, data.steps, data.box_threshold, data.mask_expand, data.mask_blur, 
            data.max_resolution_on_detection, data.sam_model_name, data.dino_model_name, data.cfg_scale,
            data.denoise, data.inpaint_padding, data.inpainting_fill, data.width, 1, data.height, 1,
            data.inpainting_mask_invert, [], data.fix_steps, True, data.sd_model_checkpoint, 'Random', [], None,
            False, [], None, False, None,
        )[0][0]

        return {"image": encode_pil_to_base64(result).decode()}


    @app.post("/replacer/avaliable_options")
    async def api_replacer_avaliable_options() -> Any:
        return {
            "sam_model_name": sam_model_list,
            "dino_model_name": dino_model_list,
            "upscalers": [""] + [x.name for x in shared.sd_upscalers],
            "lama_cleaner_avaliable": lama_cleaner_avaliable, # inpainting_fill=4, https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content
        }


script_callbacks.on_app_started(replacer_api)

