import math
from PIL import Image
import modules.shared as shared
from modules.shared import opts
from modules.images import save_image
from modules import sd_models, errors
from replacer.mask_creator import MaskResult, NothingDetectedError, createMask
from replacer.generation_args import GenerationArgs, AppropriateData
from replacer.options import EXT_NAME, needAutoUnloadModels
from replacer.tools import clearCache, interrupted, Pause
from replacer.inpaint import inpaint
from replacer.hires_fix import getGenerationArgsForHiresFixPass, prepareGenerationArgsBeforeHiresFixPass


class InterruptedDetection(Exception):
    def __init__(self):
        super().__init__("InterruptedDetection")



def generateSingle(
    image : Image.Image,
    gArgs : GenerationArgs,
    savePath : str,
    saveSuffix : str,
    save_to_dirs : bool,
    extra_includes : list,
    batch_processed : list,
):
    if interrupted():
        raise InterruptedDetection()

    maskResult: MaskResult = createMask(image, gArgs)
    gArgs.mask = maskResult.mask

    if needAutoUnloadModels():
        clearCache()

    if interrupted():
        raise InterruptedDetection()

    shared.state.assign_current_image(maskResult.maskPreview)
    shared.state.textinfo = "inpainting"

    processed, scriptImages = inpaint(image, gArgs, savePath, saveSuffix, save_to_dirs,
        batch_processed)

    extraImages = []
    if "mask" in extra_includes:
        extraImages.append(gArgs.mask)
    if "box" in extra_includes and maskResult.maskBox is not None:
        extraImages.append(maskResult.maskBox)
    if "cut" in extra_includes and maskResult.maskCut is not None:
        extraImages.append(maskResult.maskCut)
    if "preview" in extra_includes and maskResult.maskPreview is not None:
        extraImages.append(maskResult.maskPreview)
    if "script" in extra_includes:
        extraImages.extend(scriptImages)

    return processed, extraImages



def generate(
    gArgs: GenerationArgs,
    saveDir: str,
    saveToSubdirs: bool,
    useSaveFormatForVideo: bool,
    extra_includes: list,
):
    restoreList = []
    try:
        Pause.paused = False
        shared.total_tqdm.clear()
        shared.state.job_count = len(gArgs.images) * gArgs.batch_count
        totalSteps = shared.state.job_count * min(math.ceil(gArgs.steps * (1 if gArgs.img2img_fix_steps else gArgs.denoising_strength) + 1), gArgs.steps)

        if gArgs.pass_into_hires_fix_automatically:
            hiresCount = shared.state.job_count * gArgs.batch_size
            totalSteps += hiresCount * gArgs.hires_fix_args.steps
            shared.state.job_count += hiresCount
        shared.total_tqdm.updateTotal(totalSteps)

        if not gArgs.override_sd_model or gArgs.sd_model_checkpoint is None or gArgs.sd_model_checkpoint == "":
            gArgs.sd_model_checkpoint = opts.sd_model_checkpoint
        else:
            shared.state.textinfo = "switching sd checkpoint"
            oldModel = opts.sd_model_checkpoint
            def restore():
                opts.sd_model_checkpoint = oldModel
            restoreList.append(restore)
            opts.sd_model_checkpoint = gArgs.sd_model_checkpoint
            sd_models.reload_model_weights()


        if gArgs.pass_into_hires_fix_automatically:
            prepareGenerationArgsBeforeHiresFixPass(gArgs)

        n = len(gArgs.images)
        processed = None
        allExtraImages = []
        batch_processed = None


        for idx, image in enumerate(gArgs.images):
            progressInfo = "generating mask"
            if n > 1:
                print(flush=True)
                print()
                print(f'    [{EXT_NAME}]    processing {idx+1}/{n}')
                progressInfo += f" {idx+1}/{n}"
                Pause.wait()

            shared.state.textinfo = progressInfo
            shared.state.skipped = False

            if interrupted():
                if needAutoUnloadModels():
                    clearCache()
                break

            try:
                saveSuffix = ""
                if gArgs.pass_into_hires_fix_automatically:
                    saveSuffix = "-before-hires-fix"
                saveDir_ = saveDir
                if gArgs.pass_into_hires_fix_automatically and not gArgs.save_before_hires_fix:
                    saveDir_ = None
                lenImagesBefore = len(batch_processed.images) if batch_processed else 0

                processed, extraImages = generateSingle(image, gArgs, saveDir_, saveSuffix,
                    saveToSubdirs, extra_includes, batch_processed)

                if gArgs.pass_into_hires_fix_automatically:
                    hrGArgs = getGenerationArgsForHiresFixPass(gArgs)
                    for i in range(lenImagesBefore, len(processed.images)):
                        shared.state.textinfo = 'applying hires fix'
                        if interrupted():
                            break
                        processed2, _ = inpaint(processed.images[i], hrGArgs, saveDir, "", saveToSubdirs)
                        processed.images[i] = processed2.images[0]
                        processed.infotexts[i] = processed2.infotexts[0]

                for i in range(len(processed.images) - lenImagesBefore):
                    processed.images[lenImagesBefore+i].appropriateInputImageData = AppropriateData(idx, gArgs.mask, gArgs.seed+i)

            except Exception as e:
                if type(e) is InterruptedDetection:
                    break
                print(f'    [{EXT_NAME}]    Exception: {e}')
                if type(e) is not NothingDetectedError:
                    errors.report('***', exc_info=True)
                if needAutoUnloadModels():
                    clearCache()
                if n == 1:
                    raise
                if useSaveFormatForVideo:
                    save_image(image, saveDir, "", gArgs.seed, gArgs.positivePrompt,
                            opts.samples_format, save_to_dirs=False)
                shared.state.nextjob()
                continue

            allExtraImages += extraImages
            batch_processed = processed

        if processed is None:
            return None, None

        processed.info = processed.infotexts[0]

        return processed, allExtraImages

    finally:
        for restore in restoreList:
            restore()






