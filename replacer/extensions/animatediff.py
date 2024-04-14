import copy, os
from PIL import Image
from modules import scripts, shared, paths_internal
from replacer.generation_args import AnimateDiffArgs, GenerationArgs
from replacer.tools import applyMask


# --- AnimateDiff ---- https://github.com/continue-revolution/sd-webui-animatediff

SCRIPT : scripts.Script = None
AnimateDiffProcess = None


def initAnimateDiffScript():
    global SCRIPT, AnimateDiffProcess, motion_module
    index = None
    for idx, script in enumerate(scripts.scripts_img2img.alwayson_scripts):
        if script.title().lower() == "animatediff":
            index = idx
            break
    if index is not None:
        SCRIPT = copy.copy(scripts.scripts_img2img.alwayson_scripts[index])
    else:
        return
    
    if not AnimateDiffProcess:
        from scripts.animatediff_ui import AnimateDiffProcess




def apply(p, animatediff_args: AnimateDiffArgs):
    global SCRIPT, AnimateDiffProcess
    if p.script_args[SCRIPT.args_from] is None:
        p.script_args[SCRIPT.args_from] = AnimateDiffProcess()

    if not animatediff_args.needApplyAnimateDiff:
        p.script_args[SCRIPT.args_from].enable = False
    else:
        params = p.script_args[SCRIPT.args_from]
        params.enable = True
        params.format = ["PNG"]
        params.closed_loop = 'N'

        params.video_length = animatediff_args.fragment_length
        params.fps = animatediff_args.internal_fps
        params.batch_size = animatediff_args.batch_size
        params.stride = animatediff_args.stride
        params.overlap = animatediff_args.overlap
        params.latent_power = animatediff_args.latent_power
        params.latent_scale = animatediff_args.latent_scale
        params.model = animatediff_args.moution_model

        params.video_path = animatediff_args.video_path
        params.mask_path = animatediff_args.mask_path
        
        p.script_args[SCRIPT.args_from] = params


def restoreAfterCN_animatediff(gArgs: GenerationArgs, processed):
    def readImages(input_dir):
        image_list = shared.listfiles(input_dir)
        for filename in image_list:
            image = Image.open(filename).convert('RGBA')
            yield image

    newImages = []

    for res, orig, mask in \
        zip(processed.images,
            readImages(gArgs.animatediff_args.video_path),
            readImages(gArgs.animatediff_args.mask_path)
            ):
        orig = applyMask(res, orig, mask, gArgs)
        newImages.append(orig)

    processed.images = newImages



def getModels() -> list:
    if SCRIPT is None:
        return ["None"]
    models = []
    try:
        default_model_dir = os.path.join(paths_internal.extensions_dir, "sd-webui-animatediff", "model")
        model_dir = shared.opts.data.get("animatediff_model_path", default_model_dir)
        if not model_dir:
            model_dir = default_model_dir
        models = shared.listfiles(model_dir)
    except Exception as e:
        print(e)
        raise
    if models == []:
        return ["None"]
    models = [os.path.basename(x) for x in models]
    return models

