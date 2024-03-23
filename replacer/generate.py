from PIL import Image
import modules.shared as shared
from modules.shared import opts
from modules.images import save_image
from modules import sd_models
from replacer.mask_creator import MasksCreator
from replacer.generation_args import GenerationArgs
from replacer.options import EXT_NAME, needAutoUnloadModels
from replacer import replacer_scripts
from replacer.tools import clearCache, interrupted
from replacer.inpaint import inpaint
from replacer.hires_fix import getGenerationArgsForHiresFixPass, prepareGenerationArgsBeforeHiresFixPass




def generateSingle(
    image : Image,
    gArgs : GenerationArgs,
    savePath : str,
    saveSuffix : str,
    save_to_dirs : bool,
    extra_includes : list,
    batch_processed : list,
):
    maskPreview = None
    maskCutted = None
    maskBox = None

    if gArgs.use_inpaint_diff:
        gArgs.mask = replacer_scripts.InpaintDifferenceGlobals.generated_mask

    elif gArgs.only_custom_mask and gArgs.custom_mask is not None:
        gArgs.mask = gArgs.custom_mask

    else:
        masksCreator = MasksCreator(gArgs.detectionPrompt, gArgs.avoidancePrompt, image, gArgs.samModel,
            gArgs.grdinoModel, gArgs.boxThreshold, gArgs.maskExpand, gArgs.maxResolutionOnDetection,
            gArgs.avoidance_mask, gArgs.custom_mask)

        if masksCreator.previews != []:
            if gArgs.mask_num == 'Random':
                maskNum = gArgs.seed % len(masksCreator.previews)
            else:
                maskNum = int(gArgs.mask_num) - 1
            gArgs.mask = masksCreator.masks[maskNum]
            gArgs.mask_num_for_metadata = maskNum + 1

            maskPreview = masksCreator.previews[maskNum]
            maskCutted = masksCreator.cutted[maskNum]
            maskBox = masksCreator.boxes[maskNum]
        else:
            gArgs.mask = gArgs.custom_mask


    shared.state.assign_current_image(maskPreview)
    shared.state.textinfo = "inpainting"

    processed, scriptImages = inpaint(image, gArgs, savePath, saveSuffix, save_to_dirs,
        batch_processed)

    extraImages = []
    if "mask" in extra_includes:
        extraImages.append(gArgs.mask)
    if "box" in extra_includes and maskBox is not None:
        extraImages.append(maskBox)
    if "cutted" in extra_includes and maskCutted is not None:
        extraImages.append(maskCutted)
    if "preview" in extra_includes and maskPreview is not None:
        extraImages.append(maskPreview)
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
        shared.total_tqdm.clear()
        shared.state.job_count = gArgs.generationsN * gArgs.batch_count
        totalSteps = shared.state.job_count * gArgs.steps
        if gArgs.pass_into_hires_fix_automatically:
            totalSteps += shared.state.job_count * gArgs.hires_fix_args.steps
            shared.state.job_count *= 2
        shared.total_tqdm.updateTotal(totalSteps)

        if not gArgs.override_sd_model or gArgs.sd_model_checkpoint is None or gArgs.sd_model_checkpoint == "":
            gArgs.sd_model_checkpoint = opts.sd_model_checkpoint
        else:
            shared.state.textinfo = "Switching sd checkpoint"
            oldModel = opts.sd_model_checkpoint
            def restore():
                opts.sd_model_checkpoint = oldModel
            restoreList.append(restore)
            opts.sd_model_checkpoint = gArgs.sd_model_checkpoint
            sd_models.reload_model_weights()

        if useSaveFormatForVideo:
            old_samples_filename_pattern = opts.samples_filename_pattern
            old_save_images_add_number = opts.save_images_add_number
            def restoreOpts():
                opts.samples_filename_pattern = old_samples_filename_pattern
                opts.save_images_add_number = old_save_images_add_number
            restoreList.append(restoreOpts)
            opts.samples_filename_pattern = "[seed]"
            opts.save_images_add_number = True


        if gArgs.pass_into_hires_fix_automatically:
            prepareGenerationArgsBeforeHiresFixPass(gArgs)

        i = 1
        n = gArgs.generationsN
        processed = None
        allExtraImages = []
        batch_processed = None


        for image in gArgs.images:
            if interrupted():
                if needAutoUnloadModels():
                    clearCache()
                break

            progressInfo = "Generating mask"
            if n > 1: 
                print(flush=True)
                print()
                print(f'    [{EXT_NAME}]    processing {i}/{n}')
                progressInfo += f" {i}/{n}"

            shared.state.textinfo = progressInfo
            shared.state.skipped = False


            try:
                saveSuffix = ""
                if gArgs.pass_into_hires_fix_automatically:
                    saveSuffix = "-before-hires-fix"
                saveDir_ = saveDir
                if not gArgs.save_before_hires_fix:
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

            except Exception as e:
                print(f'    [{EXT_NAME}]    Exception: {e}')

                i += 1
                if needAutoUnloadModels():
                    clearCache()
                if gArgs.generationsN == 1:
                    raise
                if useSaveFormatForVideo:
                    save_image(image, saveDir, "", gArgs.seed, gArgs.positvePrompt,
                            opts.samples_format, save_to_dirs=False)
                shared.state.nextjob()
                continue

            allExtraImages += extraImages
            batch_processed = processed
            i += 1

        return processed, allExtraImages

    finally:
        for restore in restoreList:
            restore()






