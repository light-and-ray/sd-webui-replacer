import copy
from modules import scripts


# --- LamaCleaner as masked content ---- https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content


script = None

def initLamaCleanerAsMaskedContent():
    global script
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title() == "Lama-cleaner-masked-content":
            index = idx
            break
    if index is not None:
        script = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])

