import copy
import numpy as np
import gradio as gr
from PIL import ImageChops
from modules import scripts, errors
from replacer.tools import limitImageByOneDimension, applyMaskBlur, applyMask, applyRotationFix
from replacer.generation_args import GenerationArgs
from replacer.extensions.animatediff import restoreAfterCN_animatediff



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
        SCRIPT.name = scripts.scripts_img2img.alwayson_scripts[cnet_idx].name


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
def convertIntoCNImageFormat(image):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        from annotator.util import HWC3
        g_cn_HWC3 = HWC3

    color = g_cn_HWC3(np.asarray(image).astype(np.uint8))
    return color


def restoreAfterCN(origImage, mask, gArgs: GenerationArgs, processed):
    print('Restoring images resolution after ControlNet Inpainting')

    if gArgs.animatediff_args and gArgs.animatediff_args.needApplyAnimateDiff:
        restoreAfterCN_animatediff(gArgs, processed)
    else:
        origMask = mask.convert('RGBA')
        origMask = applyMaskBlur(origMask, gArgs.mask_blur)
        upscaler = gArgs.upscalerForImg2Img
        if upscaler == "":
            upscaler = None

        for i in range(len(processed.all_seeds)):
            image = applyMask(processed.images[i], origImage, origMask, gArgs)
            processed.images[i] = image


class UnitIsReserved(Exception):
    def __init__(self, unitNum: int):
        super().__init__(
            f"You have enabled ControlNet Unit {unitNum}, while it's reserved for "
            "AnimateDiff video inpainting. Please disable it. If you need more units, "
            "increase maximal number of them in Settings -> ControlNet")


def enableInpaintModeForCN(gArgs: GenerationArgs, p, previousFrame):
    if IS_SD_WEBUI_FORGE: return
    global external_code
    gArgs.cn_args = list(gArgs.cn_args)
    hasInpainting = False
    mask = None
    needApplyAnimateDiff = False
    if gArgs.animatediff_args:
        needApplyAnimateDiff = gArgs.animatediff_args.needApplyAnimateDiff

    for i in range(len(gArgs.cn_args)):
        gArgs.cn_args[i] = copy.copy(external_code.to_processing_unit(gArgs.cn_args[i]))

        if gArgs.previous_frame_into_controlnet and f"Unit {i}" in gArgs.previous_frame_into_controlnet:
            if previousFrame:
                print(f'Passing the previous frame into CN unit {i}')
                previousFrame = applyRotationFix(previousFrame, gArgs.rotation_fix)
                gArgs.cn_args[i].image = {
                    "image": convertIntoCNImageFormat(previousFrame),
                }
                gArgs.cn_args[i].enabled = True
            else:
                print(f'Disabling CN unit {i} for the first frame')
                gArgs.cn_args[i].enabled = False
                continue

        if not needApplyAnimateDiff and \
                'sparsectrl' in gArgs.cn_args[i].model.lower() and \
                gArgs.cn_args[i].enabled:
            print(f'Sparsectrl was disabled in unit {i} because of non-animatediff generation')
            gArgs.cn_args[i].enabled = False
            continue

        if gArgs.animatediff_args and gArgs.animatediff_args.needApplyCNForAnimateDiff and i+1 == len(gArgs.cn_args):
            if gArgs.cn_args[i].enabled and gArgs.cn_args[i].module != 'inpaint_only':
                raise UnitIsReserved(i)
            gArgs.cn_args[i].enabled = True
            gArgs.cn_args[i].module = 'inpaint_only'
            if gArgs.inpainting_fill > 3: # lama cleaner
                gArgs.cn_args[i].module += "+lama"
            gArgs.cn_args[i].model = gArgs.animatediff_args.cn_inpainting_model
            gArgs.cn_args[i].weight = gArgs.animatediff_args.control_weight

        if not gArgs.cn_args[i].enabled:
            continue

        if not IS_SD_WEBUI_FORGE and gArgs.cn_args[i].module.startswith('inpaint_only'):
            hasInpainting = True
            if not needApplyAnimateDiff and gArgs.originalH and gArgs.originalW:
                p.height = gArgs.originalH
                p.width = gArgs.originalW
            if p.image_mask is not None:
                mask = p.image_mask
                if p.inpainting_mask_invert:
                    mask = ImageChops.invert(mask)
                mask = applyMaskBlur(mask, p.mask_blur)

            print('Use cn inpaint instead of sd inpaint')
            image = limitImageByOneDimension(p.init_images[0], max(p.width, p.height))
            if not needApplyAnimateDiff:
                gArgs.cn_args[i].image = {
                    "image": convertIntoCNImageFormat(image),
                    "mask": convertIntoCNImageFormat(mask.resize(image.size)),
                }
            else:
                from scripts.enums import InputMode
                gArgs.cn_args[i].input_mode = InputMode.BATCH
                gArgs.cn_args[i].batch_modifiers = []
            p.image_mask = None
            p.inpaint_full_res = False
            p.needRestoreAfterCN = True


    if hasInpainting:
        for i in range(len(gArgs.cn_args)):
            gArgs.cn_args[i].inpaint_crop_input_image = False
            gArgs.cn_args[i].resize_mode = external_code.ResizeMode.OUTER_FIT



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


def getInpaintModels() -> list:
    global external_code
    if external_code is None:
        return ["None"]

    result = []
    try:
        models = []
        if IS_SD_WEBUI_FORGE:
            from lib_controlnet import global_state
            global_state.update_controlnet_filenames() 
            models = global_state.get_all_controlnet_names()
        else:
            models = external_code.get_models()
        for model in models:
            if "inpaint" in model.lower():
                result.append(model)
    except Exception as e:
        errors.report(f"{e} ***", exc_info=True)
    if result == []:
        return ["None"]
    return result

