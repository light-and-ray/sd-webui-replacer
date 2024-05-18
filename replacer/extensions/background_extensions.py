import copy
from modules import scripts


# --- Extensions which doesn't have arguments ---


SCRIPTS = None

def initAllBackgroundExtensions():
    global SCRIPTS
    SCRIPTS = []
    for script in scripts.scripts_img2img.alwayson_scripts:
        if script.args_from == script.args_to and script.args_from is not None:
            SCRIPTS.append(copy.copy(script))



# --- LamaCleaner as masked content ---- https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content

_lamaCleanerAvaliable = None

def lamaCleanerAvaliable():
    global _lamaCleanerAvaliable
    if _lamaCleanerAvaliable is None:
        _lamaCleanerAvaliable = "Lama-cleaner-masked-content" in (x.title() for x in scripts.scripts_img2img.alwayson_scripts)
    return _lamaCleanerAvaliable

