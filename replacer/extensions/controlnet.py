import copy
import numpy as np
import gradio as gr
from PIL import ImageChops
from modules import scripts, errors
from modules.images import resize_image
from replacer.tools import limitImageByOneDemention, applyMaskBlur
from replacer.generation_args import GenerationArgs



# --- ControlNet ---- https://github.com/Mikubill/sd-webui-controlnet

try:
    from lib_controlnet import external_code
    IS_SD_WEBUI_FORGE = True
except:
    external_code = None
    IS_SD_WEBUI_FORGE = False

SCRIPT : scripts.Script = None
ControlNetUiGroup = None


def initCNScript():
    global SCRIPT, ControlNetUiGroup, external_code
    cnet_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "controlnet":
            cnet_idx = idx
            break
    if cnet_idx is not None:
        SCRIPT = copy.copy(scripts.scripts_img2img.alwayson_scripts[cnet_idx])
    else:
        return
    
    try:
        if not IS_SD_WEBUI_FORGE:
            from scripts.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
            from scripts import external_code
        else:
            from lib_controlnet.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
    except:
        errors.report('Cannot register ControlNetUiGroup', exc_info=True)
        SCRIPT = None
    initCNContext()


def reinitCNScript():
    global SCRIPT
    if SCRIPT is None:
        return
    cnet_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "controlnet":
            cnet_idx = idx
            break
    if cnet_idx is not None:
        SCRIPT.args_from =  scripts.scripts_img2img.alwayson_scripts[cnet_idx].args_from
        SCRIPT.args_to =  scripts.scripts_img2img.alwayson_scripts[cnet_idx].args_to


oldCNContext = None
def initCNContext():
    global ControlNetUiGroup, oldCNContext
    oldCNContext = copy.copy(ControlNetUiGroup.a1111_context)
    ControlNetUiGroup.a1111_context.img2img_submit_button = gr.Button(visible=False)

def restoreCNContext():
    global ControlNetUiGroup, oldCNContext
    if not ControlNetUiGroup:
        return
    ControlNetUiGroup.a1111_context = copy.copy(oldCNContext)
    ControlNetUiGroup.all_ui_groups = []

g_cn_HWC3 = None
def convertIntoCNImageFromat(image):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        from annotator.util import HWC3
        g_cn_HWC3 = HWC3

    color = g_cn_HWC3(np.asarray(image).astype(np.uint8))
    return color


def restoreAfterCN(origImage, gArgs: GenerationArgs, processed):
    print('Restoring images resolution after ControlNet Inpainting')
    origMask = gArgs.mask.convert('RGBA')
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


def enableInpaintModeForCN(gArgs, p, previousFrame):
    if IS_SD_WEBUI_FORGE: return
    global external_code
    gArgs.cn_args = list(gArgs.cn_args)
    mask = None

    for i in range(len(gArgs.cn_args)):
        gArgs.cn_args[i] = external_code.to_processing_unit(gArgs.cn_args[i])

        if f"Unit {i}" in gArgs.previous_frame_into_controlnet:
            if previousFrame:
                print(f'Passing the previous frame into CN unit {i}')
                gArgs.cn_args[i].image = {
                    "image": convertIntoCNImageFromat(previousFrame),
                }
                gArgs.cn_args[i].enabled = True
            else:
                print(f'Disabling CN unit {i} for the first frame')
                gArgs.cn_args[i].enabled = False
                continue

        if not gArgs.cn_args[i].enabled:
            continue

        if not IS_SD_WEBUI_FORGE and gArgs.cn_args[i].module == 'inpaint_only':
            if p.image_mask is not None:
                mask = p.image_mask
                if p.inpainting_mask_invert:
                    mask = ImageChops.invert(mask)
                mask = applyMaskBlur(mask, p.mask_blur)

            print('Use cn inpaint instead of sd inpaint')
            image = limitImageByOneDemention(p.init_images[0], max(p.width, p.height))
            gArgs.cn_args[i].image = {
                "image": convertIntoCNImageFromat(image),
                "mask": convertIntoCNImageFromat(mask.resize(image.size)),
            }
            p.image_mask = None
            p.inpaint_full_res = False
            p.width, p.height = image.size
            gArgs.cn_args[i].inpaint_crop_input_image = False
            gArgs.cn_args[i].resize_mode = external_code.ResizeMode.RESIZE
            p.needRestoreAfterCN = True



needWatchControlNetUI = False
controlNetAccordion = None

def watchControlNetUI(component, **kwargs):
    global needWatchControlNetUI, controlNetAccordion
    if not needWatchControlNetUI:
        return

    elem_id = kwargs.get('elem_id', None)
    if elem_id is None:
        return

    if elem_id == 'controlnet':
        controlNetAccordion = component
        return

    if 'img2img' in elem_id:
        component.elem_id = elem_id.replace('img2img', 'replacer')

