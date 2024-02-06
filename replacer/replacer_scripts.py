import copy
import numpy as np
from PIL import Image, ImageChops
from modules import scripts
from modules.images import resize_image
from replacer.tools import limitSizeByOneDemention, applyMaskBlur
from replacer.generation_args import GenerationArgs

try:
    from lib_controlnet.external_code import ResizeMode
    IS_SD_WEBUI_FORGE = True
except:
    ResizeMode = None
    IS_SD_WEBUI_FORGE = False

script_controlnet : scripts.Script = None


def initCNScript():
    global script_controlnet
    cnet_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "controlnet":
            cnet_idx = idx
            break
    if cnet_idx is not None:
        script_controlnet = copy.copy(scripts.scripts_img2img.alwayson_scripts[cnet_idx])


g_cn_HWC3 = None
def convertIntoCNImageFromat(image):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        from annotator.util import HWC3
        g_cn_HWC3 = HWC3

    color = g_cn_HWC3(np.asarray(image))
    return color


def restoreAfterCN(origImage, gArgs: GenerationArgs, processed):
    print('Restoring images resolution after ControlNet Inpainting')
    origMask = gArgs.mask
    if gArgs.inpainting_mask_invert:
        origMask = ImageChops.invert(gArgs.mask)
    origMask = applyMaskBlur(origMask, gArgs.mask_blur)
    upscaler = gArgs.upscalerForImg2Img
    if upscaler == "":
        upscaler = None

    for i in range(len(processed.images)):
        imageOrg = copy.copy(origImage)
        w, h = imageOrg.size
        imageProc = resize_image(0, processed.images[i].convert('RGB'), w, h, upscaler).convert('RGBA')
        imageOrg.paste(imageProc, origMask.resize(imageOrg.size))
        processed.images[i] = imageOrg


def initResizeMode():
    global ResizeMode
    from internal_controlnet.external_code import ResizeMode


def enableInpaintModeForCN(controlNetUnits, p):
    mask = None

    for controlNetUnit in controlNetUnits:
        if not controlNetUnit.enabled:
            continue

        if not IS_SD_WEBUI_FORGE and controlNetUnit.module == 'inpaint_only':
            if p.image_mask is not None:
                mask = p.image_mask
                if p.inpainting_mask_invert:
                    mask = ImageChops.invert(mask)
                mask = applyMaskBlur(mask, p.mask_blur)
                if ResizeMode is None:
                    initResizeMode()

            print('Use cn inpaint instead of sd inpaint')
            image = limitSizeByOneDemention(p.init_images[0], max(p.width, p.height))
            controlNetUnit.image = {
                "image": convertIntoCNImageFromat(image),
                "mask": convertIntoCNImageFromat(mask.resize(image.size)),
            }
            p.image_mask = None
            p.inpaint_full_res = False
            p.width, p.height = image.size
            controlNetUnit.inpaint_crop_input_image = False
            controlNetUnit.resize_mode = ResizeMode.RESIZE
            p.needRestoreAfterCN = True
        else:
            controlNetUnit.inpaint_crop_input_image = True

    
