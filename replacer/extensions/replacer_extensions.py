import copy
from modules import scripts
from replacer.generation_args import GenerationArgs

from replacer.extensions import controlnet
from replacer.extensions import inpaint_difference
from replacer.extensions import soft_inpainting
from replacer.extensions import lama_cleaner
from replacer.extensions import image_comparison
from replacer.extensions import animatediff



def initAllScripts():
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    controlnet.initCNScript()
    inpaint_difference.initInpaintDiffirence()
    soft_inpainting.initSoftInpaintScript()
    lama_cleaner.initLamaCleanerAsMaskedContent()
    animatediff.initAnimateDiffScript()

def restoreTemporartChangedThigs():
    controlnet.restoreCNContext()

def reinitAllScreiptsAfterUICreated(*args): # for args_to and args_from
    controlnet.reinitCNScript()
    soft_inpainting.reinitSoftInpaintScript()
    lama_cleaner.initLamaCleanerAsMaskedContent()
    animatediff.initAnimateDiffScript()

def prepareScriptsArgs(scripts_args):
    result = []
    lastIndex = 0

    if controlnet.SCRIPT:
        argsLen = controlnet.SCRIPT.args_to - controlnet.SCRIPT.args_from
        result.append(scripts_args[lastIndex:lastIndex+argsLen])
        lastIndex += argsLen
    else:
        result.append([])

    if soft_inpainting.SCRIPT:
        argsLen = soft_inpainting.SCRIPT.args_to - soft_inpainting.SCRIPT.args_from
        result.append(scripts_args[lastIndex:lastIndex+argsLen])
        lastIndex += argsLen
    else:
        result.append([])
    
    return result


def applyScripts(p, gArgs: GenerationArgs):
    needControlNet = controlnet.SCRIPT is not None and gArgs.cn_args is not None and len(gArgs.cn_args) != 0
    needSoftInpaint = soft_inpainting.SCRIPT is not None and gArgs.soft_inpaint_args is not None and len(gArgs.soft_inpaint_args) != 0

    avaliableScripts = []
    if needControlNet:
        avaliableScripts.append(controlnet.SCRIPT)
    if needSoftInpaint :
        avaliableScripts.append(soft_inpainting.SCRIPT)
    if lama_cleaner.SCRIPT is not None:
        avaliableScripts.append(lama_cleaner.SCRIPT)
    if animatediff.SCRIPT is not None:
        avaliableScripts.append(animatediff.SCRIPT)

    if len(avaliableScripts) == 0:
        return

    allArgsLen = max(x.args_to for x in avaliableScripts)

    p.scripts = scripts.scripts_img2img
    p.scripts.alwayson_scripts = avaliableScripts
    p.script_args = [None] * allArgsLen

    if needControlNet:
        for i in range(len(gArgs.cn_args)):
            p.script_args[controlnet.SCRIPT.args_from + i] = gArgs.cn_args[i]
    
    if needSoftInpaint:
        for i in range(len(gArgs.soft_inpaint_args)):
            p.script_args[soft_inpainting.SCRIPT.args_from + i] = gArgs.soft_inpaint_args[i]
    
    if animatediff.SCRIPT is not None:
        animatediff.apply(p, gArgs.animatediff_args)



def prepareScriptsArgs_api(scriptsApi : dict):
    cn_args = []
    soft_inpaint_args = []

    for scriptApi in scriptsApi.items():
        if scriptApi[0] == controlnet.SCRIPT.name:
            cn_args = scriptApi[1]["args"]
            continue
        if scriptApi[0] == soft_inpainting.SCRIPT.name:
            soft_inpaint_args = scriptApi[1]["args"]
            continue
    return [cn_args, soft_inpaint_args]


def getAvaliableScripts_api():
    result = []
    if controlnet.SCRIPT:
        result.append(controlnet.SCRIPT.name)
    if soft_inpainting.SCRIPT:
        result.append(soft_inpainting.SCRIPT.name)
    return result

