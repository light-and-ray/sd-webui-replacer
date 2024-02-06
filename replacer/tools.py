from PIL import ImageChops, Image
import numpy as np
import cv2, random
from dataclasses import dataclass
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


def applyMaskBlur(mask, mask_blur):
    mask_blur_x = mask_blur
    mask_blur_y = mask_blur
    image_mask = mask
    if mask_blur_x > 0:
        np_mask = np.array(image_mask)
        kernel_size = 2 * int(2.5 * mask_blur_x + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur_x)
        image_mask = Image.fromarray(np_mask)

    if mask_blur_y > 0:
        np_mask = np.array(image_mask)
        kernel_size = 2 * int(2.5 * mask_blur_y + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur_y)
        image_mask = Image.fromarray(np_mask)
    
    return image_mask



def generateSeed():
    return int(random.randrange(4294967294))

