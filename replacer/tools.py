import os
from PIL import ImageChops, Image
from dataclasses import dataclass
import numpy as np
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from replacer.generation_args import GenerationArgs

def addReplacerMetadata(p, gArgs: GenerationArgs):
    if gArgs.detectionPrompt != '':
        p.extra_generation_params["Detection prompt"] = gArgs.detectionPrompt
    if gArgs.avoidancePrompt != '':
        p.extra_generation_params["Avoidance prompt"] = gArgs.avoidancePrompt
    p.extra_generation_params["Sam model"] = gArgs.samModel
    p.extra_generation_params["GrDino model"] = gArgs.grdinoModel
    p.extra_generation_params["Box threshold"] = gArgs.boxThreshold
    p.extra_generation_params["Mask expand"] = gArgs.maskExpand
    p.extra_generation_params["Max resolution on detection"] = gArgs.maxResolutionOnDetection
    if gArgs.mask_num_for_metadata is not None:
        p.extra_generation_params["Max num"] = gArgs.mask_num_for_metadata

def areImagesTheSame(image_one, image_two):
    if image_one is None or image_two is None:
        return image_one is None and image_two is None
    if image_one.size != image_two.size:
        return False
    diff = ImageChops.difference(image_one.convert('RGB'), image_two.convert('RGB'))

    if diff.getbbox():
        return False
    else:
        return True


def limitSizeByOneDemention(image: Image, size: int):
    if image is None:
        return None
    w, h = image.size
    if h > w:
        if h > size:
            w = size / h * w
            h = size
    else:
        if w > size:
            h = size / w * h
            w = size

    return image.resize((int(w), int(h)))



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
        raise Exception("Not an image")


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image).decode()
    elif type(image) is np.ndarray:
        pil = Image.fromarray(image)
        return encode_pil_to_base64(pil).decode()
    else:
        raise Exception("Invalid type")


@dataclass
class CashedExtraMaskExpand:
    mask: Image
    expand: int
    result: Image
cashedExtraMaskExpand: CashedExtraMaskExpand = None

update_mask = None

def extraMaskExpand(mask: Image, expand: int):
    global cashedExtraMaskExpand, update_mask

    if cashedExtraMaskExpand is not None and\
            cashedExtraMaskExpand.expand == expand and\
            areImagesTheSame(cashedExtraMaskExpand.mask, mask):
        print('extraMaskExpand restored from cache')
        return cashedExtraMaskExpand.result
    else:
        if update_mask is None:
            from scripts.sam import update_mask as update_mask_
            update_mask = update_mask_
        expandedMask = update_mask(mask, 0, expand, mask.convert('RGBA'))[1]
        cashedExtraMaskExpand = CashedExtraMaskExpand(mask, expand, expandedMask)
        print('extraMaskExpand cached')
        return expandedMask


def prepareMask(mask_mode, mask_raw):
    if mask_mode is None or mask_raw is None:
        return None
    mask = None
    if 'Upload mask' in mask_mode:
        mask = mask_raw['image'].convert('L')
    if 'Draw mask' in mask_mode:
        mask = Image.new('L', mask_raw['mask'].size, 0) if mask is None else mask
        draw_mask = mask_raw['mask'].convert('L')
        mask.paste(draw_mask, draw_mask)
        blackFilling = Image.new('L', mask.size, 0)
        if areImagesTheSame(blackFilling, mask):
            return None
    return mask
