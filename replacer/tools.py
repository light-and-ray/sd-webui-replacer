from PIL import ImageChops, Image
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


def prepareAvoidanceMask(avoidance_mask_mode, avoidance_mask):
    if avoidance_mask_mode is None or avoidance_mask is None:
        return None
    mask = None
    if 'Upload mask' in avoidance_mask_mode:
        mask = avoidance_mask['image'].convert('L')
    if 'Draw mask' in avoidance_mask_mode:
        mask = Image.new('L', avoidance_mask['mask'].size, 0) if mask is None else mask
        draw_mask = avoidance_mask['mask'].convert('L')
        mask.paste(draw_mask, draw_mask)
    return mask


def prepareCustomMask(custom_mask_mode, custom_mask):
    if custom_mask_mode is None or custom_mask is None:
        return None
    mask = None
    if 'Upload mask' in custom_mask_mode:
        mask = custom_mask['image'].convert('L')
    if 'Draw mask' in custom_mask_mode:
        mask = Image.new('L', custom_mask['mask'].size, 0) if mask is None else mask
        draw_mask = custom_mask['mask'].convert('L')
        mask.paste(draw_mask, draw_mask)
        blackFilling = Image.new('L', mask.size, 0)
        if areImagesTheSame(blackFilling, mask):
            return None
    return mask
