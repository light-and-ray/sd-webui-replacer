import copy
from modules import scripts



# --- ArPlusPlus ---- https://github.com/altoiddealer/--sd-webui-ar-plusplus (maybe works with other forks)

SCRIPT : scripts.Script = None

def initArPlusPlusScript():
    global SCRIPT
    script_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Aspect Ratio picker":
            script_idx = idx
            break
    if script_idx is not None:
        SCRIPT = copy.copy(scripts.scripts_img2img.alwayson_scripts[script_idx])
    else:
        return


def reinitArPlusPlusScript():
    global SCRIPT
    script_idx = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Aspect Ratio picker":
            script_idx = idx
            break
    if script_idx is not None:
        SCRIPT = copy.copy(scripts.scripts_img2img.alwayson_scripts[script_idx])
    else:
        return
    if script_idx is not None:
        SCRIPT.args_from =  scripts.scripts_img2img.alwayson_scripts[script_idx].args_from
        SCRIPT.args_to =  scripts.scripts_img2img.alwayson_scripts[script_idx].args_to
        SCRIPT.name = scripts.scripts_img2img.alwayson_scripts[script_idx].name

