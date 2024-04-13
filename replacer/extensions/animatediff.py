import copy
from modules import scripts
from replacer.generation_args import AnimateDiffArgs


# --- AnimateDiff ---- https://github.com/continue-revolution/sd-webui-animatediff

SCRIPT : scripts.Script = None


def initAnimateDiffScript():
    global SCRIPT
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "animatediff":
            index = idx
            break
    if index is not None:
        SCRIPT = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])


def apply(p, animatediff_args: AnimateDiffArgs):
    global SCRIPT
    from scripts.animatediff_ui import AnimateDiffProcess
    params = AnimateDiffProcess()
    params.enable = True
    params.batch_size = 16
    params.video_length = 16
    params.fps = 16
    params.model = 'mm_sd15_v3.safetensors'
    params.format = ["PNG"]
    params.video_path = animatediff_args.video_path
    params.mask_path = animatediff_args.mask_path
    # params.latent_power = 1
    params.latent_scale = 15
    p.script_args[SCRIPT.args_from] = params

