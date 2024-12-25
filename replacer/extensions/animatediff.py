import copy, os
from PIL import Image
from modules import scripts, shared, paths_internal, errors, extensions
from replacer.generation_args import AnimateDiffArgs, GenerationArgs
from replacer.tools import applyMask


# --- AnimateDiff ---- https://github.com/continue-revolution/sd-webui-animatediff

SCRIPT : scripts.Script = None
AnimateDiffProcess = None


def initAnimateDiffScript():
    global SCRIPT, AnimateDiffProcess
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

    if not animatediff_args or not animatediff_args.needApplyAnimateDiff:
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
        params.freeinit_enable = animatediff_args.freeinit_enable
        params.freeinit_filter = animatediff_args.freeinit_filter
        params.freeinit_ds = animatediff_args.freeinit_ds
        params.freeinit_dt = animatediff_args.freeinit_dt
        params.freeinit_iters = animatediff_args.freeinit_iters
        params.model = animatediff_args.motion_model

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
    i = 0
    total = len(processed.all_seeds)

    for res, orig, mask in \
        zip(processed.images[:total],
            readImages(gArgs.animatediff_args.video_path),
            readImages(gArgs.animatediff_args.mask_path)
            ):
        if gArgs.upscalerForImg2Img not in [None, "None", "Lanczos", "Nearest"]:
            print(f"{i+1} / {total}")
        orig = applyMask(res, orig, mask, gArgs)
        newImages.append(orig)
        i+=1

    processed.images = newImages



def getModels() -> list:
    if SCRIPT is None:
        return ["None"]
    models = []
    try:
        try:
            adExtension = next(x for x in extensions.extensions if "animatediff" in x.name.lower())
            default_model_dir = os.path.join(adExtension.path, "model")
        except Exception as e:
            errors.report(e)
            default_model_dir = os.path.join(paths_internal.extensions_dir, "sd-webui-animatediff", "model")
        model_dir = shared.opts.data.get("animatediff_model_path", default_model_dir)
        if not model_dir:
            model_dir = default_model_dir
        models = shared.listfiles(model_dir)
        models = [os.path.basename(x) for x in models]
    except Exception as e:
        errors.report(e)

    if models == []:
        return ["None"]
    return models

