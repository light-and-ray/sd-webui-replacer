import copy
from modules import scripts

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


def enableInpaintModeForCN(args):
    from internal_controlnet.external_code import get_all_units_from
    controlNetUnits = get_all_units_from(args)
    for controlNetUnit in controlNetUnits:
        controlNetUnit.inpaint_crop_input_image = True
    
