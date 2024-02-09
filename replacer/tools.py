from PIL import ImageChops, Image
import numpy as np
import cv2, random, git
from dataclasses import dataclass
from modules import errors
from replacer.generation_args import GenerationArgs

try:
    REPLACER_VERSION = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
except Exception:
    errors.report(f"Error reading replacer git info from {__file__}", exc_info=True)
    REPLACER_VERSION = "None"



def addReplacerMetadata(p, gArgs: GenerationArgs):
    p.extra_generation_params["Extension"] = f'sd-webui-replacer {REPLACER_VERSION}'
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
        p.extra_generation_params["Mask num"] = gArgs.mask_num_for_metadata

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
class CachedExtraMaskExpand:
    mask: Image
    expand: int
    result: Image
cachedExtraMaskExpand: CachedExtraMaskExpand = None

update_mask = None

def extraMaskExpand(mask: Image, expand: int):
    global cachedExtraMaskExpand, update_mask

    if cachedExtraMaskExpand is not None and\
            cachedExtraMaskExpand.expand == expand and\
            areImagesTheSame(cachedExtraMaskExpand.mask, mask):
        print('extraMaskExpand restored from cache')
        return cachedExtraMaskExpand.result
    else:
        if update_mask is None:
            from scripts.sam import update_mask as update_mask_
            update_mask = update_mask_
        expandedMask = update_mask(mask, 0, expand, mask.convert('RGBA'))[1]
        cachedExtraMaskExpand = CachedExtraMaskExpand(mask, expand, expandedMask)
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


def applyMaskBlur(image_mask, mask_blur):
    if mask_blur > 0:
        np_mask = np.array(image_mask)
        kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)
        image_mask = Image.fromarray(np_mask)
    return image_mask



def generateSeed():
    return int(random.randrange(4294967294))

