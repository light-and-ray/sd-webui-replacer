import copy, importlib
import numpy as np
import gradio as gr
from PIL import ImageChops
from modules import scripts, errors
from modules.images import resize_image
from replacer.tools import limitSizeByOneDemention, applyMaskBlur
from replacer.generation_args import GenerationArgs
from replacer.options import EXT_NAME


# --- ControlNet ---- https://github.com/Mikubill/sd-webui-controlnet

try:
    from lib_controlnet import external_code
    IS_SD_WEBUI_FORGE = True
except:
    external_code = None
    IS_SD_WEBUI_FORGE = False

script_controlnet : scripts.Script = None
ControlNetUiGroup = None


def initCNScript():
    global script_controlnet, ControlNetUiGroup, external_code
    cnet_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "controlnet":
            cnet_idx = idx
            break
    if cnet_idx is not None:
        script_controlnet = copy.copy(scripts.scripts_img2img.alwayson_scripts[cnet_idx])
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
        script_controlnet = None
    initCNContext()


def reinitCNScript():
    global script_controlnet
    if script_controlnet is None:
        return
    cnet_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "controlnet":
            cnet_idx = idx
            break
    if cnet_idx is not None:
        script_controlnet.args_from =  scripts.scripts_img2img.alwayson_scripts[cnet_idx].args_from
        script_controlnet.args_to =  scripts.scripts_img2img.alwayson_scripts[cnet_idx].args_to


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
                print(f'Passing the previous frame CN unit {i}')
                gArgs.cn_args[i].image = {
                    "image": convertIntoCNImageFromat(previousFrame),
                    # "mask": convertIntoCNImageFromat(p.image_mask),
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
            image = limitSizeByOneDemention(p.init_images[0], max(p.width, p.height))
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
    

# --- InpaintDifference ---- https://github.com/John-WL/sd-webui-inpaint-difference

InpaintDifferenceGlobals = None
computeInpaintDifference = None

def initInpaintDiffirence():
    global InpaintDifferenceGlobals, computeInpaintDifference
    try:
        from lib_inpaint_difference.globals import DifferenceGlobals as InpaintDifferenceGlobals
    except:
        InpaintDifferenceGlobals = None
        return

    try:
        from lib_inpaint_difference.mask_processing import compute_mask
        def computeInpaintDifference(
            non_altered_image_for_inpaint_diff,
            image,
            mask_blur,
            mask_expand,
            erosion_amount,
            inpaint_diff_threshold,
            inpaint_diff_contours_only,
        ):
            if image is None or non_altered_image_for_inpaint_diff is None:
                return None
            return compute_mask(
                non_altered_image_for_inpaint_diff.convert('RGB'),
                image.convert('RGB'),
                mask_blur,
                mask_expand,
                erosion_amount,
                inpaint_diff_threshold,
                inpaint_diff_contours_only,
            )
            
    except Exception as e:
        errors.report(f"Cannot init InpaintDiffirence {e}", exc_info=True)
        InpaintDifferenceGlobals = None



# --- SoftInpainting ----

script_soft_inpaint : scripts.Script = None


def initSoftInpaintScript():
    global script_soft_inpaint
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Soft Inpainting":
            index = idx
            break
    if index is not None:
        script_soft_inpaint = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])


def reinitSoftInpaintScript():
    global script_soft_inpaint
    if script_soft_inpaint is None:
        return
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Soft Inpainting":
            index = idx
            break
    if index is not None:
        script_soft_inpaint.args_from = scripts.scripts_img2img.alwayson_scripts[index].args_from
        script_soft_inpaint.args_to = scripts.scripts_img2img.alwayson_scripts[index].args_to




needWatchSoftInpaintUI = False


def watchSoftInpaintUI(component, **kwargs):
    global needWatchSoftInpaintUI
    if not needWatchSoftInpaintUI:
        return

    elem_id = kwargs.get('elem_id', None)
    if elem_id is None:
        return

    if 'soft' in elem_id:
        component.elem_id = elem_id.replace('soft', 'replacer_soft')



# --- LamaCleaner as masked content ---- https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content


script_lama_cleaner_as_masked_content = None

def initLamaCleanerAsMaskedContent():
    global script_lama_cleaner_as_masked_content
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Lama-cleaner-masked-content":
            index = idx
            break
    if index is not None:
        script_lama_cleaner_as_masked_content = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])



# --- ImageComparison ---- https://github.com/Haoming02/sd-webui-image-comparison


def addButtonIntoComparisonTab(component, **kwargs):
    elem_id = kwargs.get('elem_id', None)
    if elem_id == 'img_comp_extras':
        column = component.parent
        with column.parent:
            with column:
                replacer_btn = gr.Button(f'Compare {EXT_NAME}', elem_id='img_comp_replacer')
        replacer_btn.click(None, None, None, _js='replacer_imageComparisonloadImage')


needWatchImageComparison = False

def watchImageComparison(component, **kwargs):
    global needWatchImageComparison
    if not needWatchImageComparison:
        return
    elem_id = kwargs.get('elem_id', None)
    if elem_id in ['img_comp_i2i', 'img_comp_inpaint', 'img_comp_extras']:
        component.visible = False


ImageComparisonTab = None

def preloadImageComparisonTab():
    global ImageComparisonTab, needWatchImageComparison
    try:
        img_comp = importlib.import_module('extensions.sd-webui-image-comparison.scripts.img_comp')
    except ImportError:
        return
    needWatchImageComparison = True
    ImageComparisonTab = img_comp.img_ui()[0]

def mountImageComparisonTab():
    global ImageComparisonTab, needWatchImageComparison
    if not ImageComparisonTab:
        return
    gr.Radio(value="Off", elem_id="setting_comp_send_btn", choices=["Off", "Text", "Icon"], visible=False)
    gr.Textbox(elem_id="replacer_image_comparison", visible=False)
    interface, label, ifid = ImageComparisonTab
    with gr.Tab(label=label, elem_id=f"tab_{ifid}"):
        interface.render()
    needWatchImageComparison = False


# --------


def initAllScripts():
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    initCNScript()
    initInpaintDiffirence()
    initSoftInpaintScript()
    initLamaCleanerAsMaskedContent()

def restoreTemporartChangedThigs():
    restoreCNContext()

def reinitAllScreiptsAfterUICreated(*args):
    reinitCNScript()
    reinitSoftInpaintScript()
    initLamaCleanerAsMaskedContent()

def prepareScriptsArgs(scripts_args):
    global script_controlnet, script_soft_inpaint

    result = []
    lastIndex = 0

    if script_controlnet:
        argsLen = script_controlnet.args_to - script_controlnet.args_from
        result.append(scripts_args[lastIndex:lastIndex+argsLen])
        lastIndex += argsLen
    else:
        result.append([])

    if script_soft_inpaint:
        argsLen = script_soft_inpaint.args_to - script_soft_inpaint.args_from
        result.append(scripts_args[lastIndex:lastIndex+argsLen])
        lastIndex += argsLen
    else:
        result.append([])
    
    return result


def applyScripts(p, cn_args, soft_inpaint_args):
    global script_controlnet, script_soft_inpaint, script_lama_cleaner_as_masked_content
    needControlNet = script_controlnet is not None and cn_args is not None and len(cn_args) != 0
    needSoftInpaint = script_soft_inpaint is not None and soft_inpaint_args is not None and len(soft_inpaint_args) != 0

    avaliableScripts = []
    if needControlNet:
        avaliableScripts.append(script_controlnet)
    if needSoftInpaint :
        avaliableScripts.append(script_soft_inpaint)
    if script_lama_cleaner_as_masked_content is not None:
        avaliableScripts.append(script_lama_cleaner_as_masked_content)

    if len(avaliableScripts) == 0:
        return

    allArgsLen = max(x.args_to for x in avaliableScripts)

    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = avaliableScripts
    p.script_args = [None] * allArgsLen

    if needControlNet:
        for i in range(len(cn_args)):
            p.script_args[script_controlnet.args_from + i] = cn_args[i]
    
    if needSoftInpaint:
        for i in range(len(soft_inpaint_args)):
            p.script_args[script_soft_inpaint.args_from + i] = soft_inpaint_args[i]


def prepareScriptsArgs_api(scriptsApi : dict):
    global script_controlnet, script_soft_inpaint
    cn_args = []
    soft_inpaint_args = []

    for scriptApi in scriptsApi.items():
        if scriptApi[0] == script_controlnet.name:
            cn_args = scriptApi[1]["args"]
            continue
        if scriptApi[0] == script_soft_inpaint.name:
            soft_inpaint_args = scriptApi[1]["args"]
            continue
    return [cn_args, soft_inpaint_args]


def getAvaliableScripts_api():
    global script_controlnet, script_soft_inpaint
    result = []
    if script_controlnet:
        result.append(script_controlnet.name)
    if script_soft_inpaint:
        result.append(script_soft_inpaint.name)
    return result

