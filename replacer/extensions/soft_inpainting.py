import copy
from modules import scripts


# --- SoftInpainting ----

SCRIPT : scripts.Script = None


def initSoftInpaintScript():
    global SCRIPT
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Soft Inpainting":
            index = idx
            break
    if index is not None:
        SCRIPT = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])


def reinitSoftInpaintScript():
    global SCRIPT
    if SCRIPT is None:
        return
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Soft Inpainting":
            index = idx
            break
    if index is not None:
        SCRIPT.args_from = scripts.scripts_img2img.alwayson_scripts[index].args_from
        SCRIPT.args_to = scripts.scripts_img2img.alwayson_scripts[index].args_to
        SCRIPT.name = scripts.scripts_img2img.alwayson_scripts[index].name




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

