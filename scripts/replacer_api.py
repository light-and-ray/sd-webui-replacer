import os
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
from fastapi import FastAPI, Body
from pydantic import BaseModel, validator
from replacer.generate import generateSingle
from replacer.generation_args import GenerationArgs
from replacer.options import (getDetectionPromptExamples, getAvoidancePromptExamples, getPositivePromptExamples,
                              getNegativePromptExamples, useFirstPositivePromptFromExamples,
                              useFirstNegativePromptFromExamples,
                              )

from modules.api.api import encode_pil_to_base64, decode_base64_to_image


def decode_to_pil(image):
    if os.path.exists(image):
        return Image.open(image)
    elif type(image) is str:
        return decode_base64_to_image(image)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        Exception("Not an image")


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image).decode()
    elif type(image) is np.ndarray:
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    else:
        Exception("Invalid type")


def replacer_api(_: gr.Blocks, app: FastAPI):
    from scripts.sam import sam_model_list

    class ReplaceRequest(BaseModel):
        input_image: str
        detection_prompt: str = ""
        avoidance_prompt: str = ""
        positive_prompt: str = ""
        negative_prompt: str = ""
        width: int = 512
        height: int = 512
        sam_model_name: str = sam_model_list[0] if sam_model_list else ""
        dino_model_name: str = "GroundingDINO_SwinT_OGC (694MB)"
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

        @validator("detection_prompt", "avoidance_prompt", "positive_prompt", "negative_prompt")
        def validate_prompts(cls, value: str, field) -> str:
            value = value.strip()
            if value:
                return value

            if field.name == "detection_prompt":
                return getDetectionPromptExamples()[0]
            if field.name == "avoidance_prompt":
                return getAvoidancePromptExamples()[0]
            if field.name == "positive_prompt" and useFirstPositivePromptFromExamples():
                return getPositivePromptExamples()[0]
            if field.name == "negative_prompt" and useFirstNegativePromptFromExamples():
                return getNegativePromptExamples()[0]

    @app.post("/replacer/replace")
    async def api_replacer_replace(data: ReplaceRequest = Body(...)) -> Any:
        image = decode_to_pil(data.input_image).convert("RGBA")
        args = GenerationArgs(data.positive_prompt, data.negative_prompt, data.detection_prompt, data.avoidance_prompt,
                              None, None, data.seed, data.sam_model_name, data.dino_model_name, data.box_threshold,
                              data.mask_expand, data.max_resolution_on_detection, data.steps, data.sampler,
                              data.mask_blur, 1, 1, 1, data.cfg_scale, data.denoise, data.height, data.width,
                              data.inpaint_padding, False, data.inpainting_mask_invert, [image], 1, False, [])
        processed, _ = generateSingle(image, args, "", "", False, [], [])

        return {"image": encode_to_base64(processed.images[0])}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(replacer_api)
except:
    print("SAM Web UI API failed to initialize")
