import copy
from modules import scripts


# --- SoftInpainting ----

script : scripts.Script = None


def initSoftInpaintScript():
    global script
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Soft Inpainting":
            index = idx
            break
    if index is not None:
        script = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])


def reinitSoftInpaintScript():
    global script
    if script is None:
        return
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Soft Inpainting":
            index = idx
            break
    if index is not None:
        script.args_from = scripts.scripts_img2img.alwayson_scripts[index].args_from
        script.args_to = scripts.scripts_img2img.alwayson_scripts[index].args_to




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

