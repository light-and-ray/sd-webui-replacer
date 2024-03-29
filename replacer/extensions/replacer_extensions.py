import copy
from modules import scripts

from replacer.extensions import controlnet
from replacer.extensions import inpaint_difference
from replacer.extensions import soft_inpainting
from replacer.extensions import lama_cleaner
from replacer.extensions import image_comparison



def initAllScripts():
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    controlnet.initCNScript()
    inpaint_difference.initInpaintDiffirence()
    soft_inpainting.initSoftInpaintScript()
    lama_cleaner.initLamaCleanerAsMaskedContent()

def restoreTemporartChangedThigs():
    controlnet.restoreCNContext()

def reinitAllScreiptsAfterUICreated(*args):
    controlnet.reinitCNScript()
    soft_inpainting.reinitSoftInpaintScript()
    lama_cleaner.initLamaCleanerAsMaskedContent()

def prepareScriptsArgs(scripts_args):
    result = []
    lastIndex = 0

    if controlnet.script:
        argsLen = controlnet.script.args_to - controlnet.script.args_from
        result.append(scripts_args[lastIndex:lastIndex+argsLen])
        lastIndex += argsLen
    else:
        result.append([])

    if soft_inpainting.script:
        argsLen = soft_inpainting.script.args_to - soft_inpainting.script.args_from
        result.append(scripts_args[lastIndex:lastIndex+argsLen])
        lastIndex += argsLen
    else:
        result.append([])
    
    return result


def applyScripts(p, cn_args, soft_inpaint_args):
    needControlNet = controlnet.script is not None and cn_args is not None and len(cn_args) != 0
    needSoftInpaint = soft_inpainting.script is not None and soft_inpaint_args is not None and len(soft_inpaint_args) != 0

    avaliableScripts = []
    if needControlNet:
        avaliableScripts.append(controlnet.script)
    if needSoftInpaint :
        avaliableScripts.append(soft_inpainting.script)
    if lama_cleaner.script is not None:
        avaliableScripts.append(lama_cleaner.script)

    if len(avaliableScripts) == 0:
        return

    allArgsLen = max(x.args_to for x in avaliableScripts)

    p.scripts = copy.copy(scripts.scripts_img2img)
    p.scripts.alwayson_scripts = avaliableScripts
    p.script_args = [None] * allArgsLen

    if needControlNet:
        for i in range(len(cn_args)):
            p.script_args[controlnet.script.args_from + i] = cn_args[i]
    
    if needSoftInpaint:
        for i in range(len(soft_inpaint_args)):
            p.script_args[soft_inpainting.script.args_from + i] = soft_inpaint_args[i]


def prepareScriptsArgs_api(scriptsApi : dict):
    cn_args = []
    soft_inpaint_args = []

    for scriptApi in scriptsApi.items():
        if scriptApi[0] == controlnet.script.name:
            cn_args = scriptApi[1]["args"]
            continue
        if scriptApi[0] == soft_inpainting.script.name:
            soft_inpaint_args = scriptApi[1]["args"]
            continue
    return [cn_args, soft_inpaint_args]


def getAvaliableScripts_api():
    result = []
    if controlnet.script:
        result.append(controlnet.script.name)
    if soft_inpainting.script:
        result.append(soft_inpainting.script.name)
    return result

