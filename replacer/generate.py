import os, copy
from contextlib import closing
from PIL import Image
import modules.shared as shared
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts
from modules.ui import plaintext_to_html
from modules.images import save_image
from modules import scripts, sd_models, errors
from replacer.mask_creator import MasksCreator
from replacer.generation_args import GenerationArgs, HiresFixCacheData
from replacer.video_tools import getVideoFrames, save_video
from replacer.options import ( getDetectionPromptExamples, getPositivePromptExamples,
    getNegativePromptExamples, useFirstPositivePromptFromExamples, useFirstNegativePromptFromExamples,
    getHiresFixPositivePromptSuffixExamples, EXT_NAME, EXT_NAME_LOWER, getSaveDir, needAutoUnloadModels,
)
from replacer import replacer_scripts
from replacer.tools import ( addReplacerMetadata, extraMaskExpand, prepareMask, generateSeed
)

g_clear_cache = None

def clearCache():
    global g_clear_cache
    if g_clear_cache is None:
        from scripts.sam import clear_cache
        g_clear_cache = clear_cache
    g_clear_cache()


def inpaint(
    image : Image,
    gArgs : GenerationArgs,
    savePath : str = "",
    saveSuffix : str = "",
    save_to_dirs : bool = True,
    batch_processed : Processed = None
):
    override_settings = {}
    if (gArgs.upscalerForImg2Img is not None and gArgs.upscalerForImg2Img != "" and gArgs.upscalerForImg2Img != "None"):
        override_settings["upscaler_for_img2img"] = gArgs.upscalerForImg2Img
    if (gArgs.sd_model_checkpoint is not None and gArgs.sd_model_checkpoint != ""):
        override_settings["sd_model_checkpoint"] = gArgs.sd_model_checkpoint
    override_settings["img2img_fix_steps"] = gArgs.img2img_fix_steps

    inpainting_fill = gArgs.inpainting_fill
    mask = gArgs.mask.resize(image.size).convert('L')
    if (inpainting_fill == 4): # lama cleaner (https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content)
        inpainting_fill = 1 # original
        try:
            from lama_cleaner_masked_content.inpaint import lamaInpaint
            from lama_cleaner_masked_content.options import getUpscaler
            image = lamaInpaint(image, mask, gArgs.inpainting_mask_invert, getUpscaler())
        except Exception as e:
            errors.report(f'[{EXT_NAME}]: {e}', exc_info=True)

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=gArgs.positvePrompt,
        negative_prompt=gArgs.negativePrompt,
        styles=[],
        sampler_name=gArgs.sampler_name,
        batch_size=gArgs.batch_size,
        n_iter=gArgs.n_iter,
        steps=gArgs.steps,
        cfg_scale=gArgs.cfg_scale,
        width=gArgs.width,
        height=gArgs.height,
        init_images=[image],
        mask=mask,
        mask_blur=gArgs.mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=0,
        denoising_strength=gArgs.denoising_strength,
        image_cfg_scale=1.5,
        inpaint_full_res=True,
        inpaint_full_res_padding=gArgs.inpaint_full_res_padding,
        inpainting_mask_invert=gArgs.inpainting_mask_invert,
        override_settings=override_settings,
        do_not_save_samples=True,
    )

    p.extra_generation_params["Mask blur"] = gArgs.mask_blur
    addReplacerMetadata(p, gArgs)
    p.seed = gArgs.seed
    p.do_not_save_grid = True
    try:
        if replacer_scripts.script_controlnet and gArgs.cn_args is not None and len(gArgs.cn_args) != 0:
            replacer_scripts.enableInpaintModeForCN(gArgs.cn_args, p)
    except Exception as e:
        errors.report(f"Error {e}", exc_info=True)
    replacer_scripts.applyScripts(p, gArgs.cn_args, gArgs.soft_inpaint_args)



    with closing(p):
        processed = process_images(p)

    scriptImages = processed.images[len(processed.all_seeds):]
    processed.images = processed.images[:len(processed.all_seeds)]
    scriptImages.extend(getattr(processed, 'extra_images', []))

    needRestoreAfterCN = getattr(p, 'needRestoreAfterCN', False)
    if needRestoreAfterCN:
        replacer_scripts.restoreAfterCN(image, gArgs, processed)


    if savePath != "":
        for i in range(len(processed.images)):
            additional_save_suffix = getattr(image, 'additional_save_suffix', None)
            suffix = saveSuffix
            if additional_save_suffix:
                suffix = additional_save_suffix + suffix
            save_image(processed.images[i], savePath, "", processed.all_seeds[i], gArgs.positvePrompt, opts.samples_format,
                    info=processed.infotext(p, i), p=p, suffix=suffix, save_to_dirs=save_to_dirs)

    if opts.do_not_show_images:
        processed.images = []

    if batch_processed:
        batch_processed.images += processed.images
        batch_processed.infotexts += processed.infotexts
        processed = batch_processed

    return processed, scriptImages



lastGenerationArgs = None

def getLastUsedSeed():
    if lastGenerationArgs is None:
        return -1
    else:
        return lastGenerationArgs.seed



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
    shared.state.textinfo = "inpaint"

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
    detectionPrompt: str,
    avoidancePrompt: str,
    positvePrompt: str,
    negativePrompt: str,
    tab_index,
    image_single,
    image_batch,
    keep_original_filenames,
    input_batch_dir,
    output_batch_dir,
    keep_original_filenames_from_dir,
    show_batch_dir_results,
    input_video,
    video_output_dir,
    target_video_fps,
    upscalerForImg2Img,
    seed,
    sampler,
    steps,
    box_threshold,
    mask_expand,
    mask_blur,
    max_resolution_on_detection,
    sam_model_name,
    dino_model_name,
    cfg_scale,
    denoise,
    inpaint_padding,
    inpainting_fill,
    width,
    batch_count,
    height,
    batch_size,
    inpainting_mask_invert,
    extra_includes,
    fix_steps,
    override_sd_model,
    sd_model_checkpoint,
    mask_num,
    avoidance_mask_mode,
    avoidance_mask,
    only_custom_mask,
    custom_mask_mode,
    custom_mask,
    use_inpaint_diff,
    inpaint_diff_mask_view,
    *scripts_args,
):
    restoreList = []
    try:
        shared.state.begin(job=EXT_NAME_LOWER)
        shared.total_tqdm.clear()

        if detectionPrompt == '':
            detectionPrompt = getDetectionPromptExamples()[0]

        if positvePrompt == '' and useFirstPositivePromptFromExamples():
            positvePrompt = getPositivePromptExamples()[0]

        if negativePrompt == '' and useFirstNegativePromptFromExamples():
            negativePrompt = getNegativePromptExamples()[0]

        if (seed == -1):
            seed = generateSeed()

        detectionPrompt = detectionPrompt.strip()
        avoidancePrompt = avoidancePrompt.strip()
        output_batch_dir = output_batch_dir.strip()
        video_output_dir = video_output_dir.strip()

        if not override_sd_model or sd_model_checkpoint is None or sd_model_checkpoint == "":
            sd_model_checkpoint = opts.sd_model_checkpoint
        else:
            oldModel = opts.sd_model_checkpoint
            def restore():
                opts.sd_model_checkpoint = oldModel
            restoreList.append(restore)
            opts.sd_model_checkpoint = sd_model_checkpoint
            sd_models.reload_model_weights()

        images = []

        if tab_index == 0:
            if image_single is None:
                generationsN = 0
            else:
                images = [image_single]
                generationsN = 1


        if tab_index == 1:
            def getImages(image_folder):
                for img in image_folder:
                    if isinstance(img, Image.Image):
                        image = img
                    else:
                        filename = os.path.abspath(img.name)
                        image = Image.open(filename).convert('RGBA')
                        if keep_original_filenames:
                            image.additional_save_suffix = '-' + os.path.basename(filename)
                    yield image
            if image_batch is None:
                generationsN = 0
            else:
                images = getImages(image_batch)
                generationsN = len(image_batch)


        if tab_index == 2:
            def readImages(input_dir):
                image_list = shared.listfiles(input_dir)
                for filename in image_list:
                    try:
                        image = Image.open(filename).convert('RGBA')
                        if keep_original_filenames_from_dir:
                            image.additional_save_suffix = '-' + os.path.basename(filename)
                    except Exception:
                        continue
                    yield image
            images = readImages(input_batch_dir)
            generationsN = len(shared.listfiles(input_batch_dir))


        if tab_index == 3:
            shared.state.textinfo = 'video preparing'
            temp_batch_folder = os.path.join(os.path.dirname(input_video), 'temp')
            if video_output_dir == "":
                video_output_dir = os.path.join(os.path.dirname(input_video), f'out_{seed}')
            else:
                video_output_dir = os.path.join(video_output_dir, f'out_{seed}')
            if os.path.exists(video_output_dir):
                for file in os.listdir(video_output_dir):
                    if file.endswith(f'.{shared.opts.samples_format}'):
                        os.remove(os.path.join(video_output_dir, file))
            images, fps_in, fps_out = getVideoFrames(input_video, target_video_fps)
            generationsN = len(shared.listfiles(temp_batch_folder))

            batch_count = 1
            batch_size = 1
            extra_includes = []
            old_samples_filename_pattern = opts.samples_filename_pattern
            old_save_images_add_number = opts.save_images_add_number
            def restoreOpts():
                opts.samples_filename_pattern = old_samples_filename_pattern
                opts.save_images_add_number = old_save_images_add_number
            restoreList.append(restoreOpts)
            opts.samples_filename_pattern = "[seed]"
            opts.save_images_add_number = True


        if generationsN == 0:
            return [], "", plaintext_to_html("no input images"), ""
        shared.state.job_count = generationsN*batch_count

        gArgs = GenerationArgs(
            positvePrompt,
            negativePrompt,
            detectionPrompt,
            avoidancePrompt,
            None,
            upscalerForImg2Img,
            seed,
            sam_model_name,
            dino_model_name,
            box_threshold,
            mask_expand,
            max_resolution_on_detection,
            
            steps,
            sampler,
            mask_blur,
            inpainting_fill,
            batch_count,
            batch_size,
            cfg_scale,
            denoise,
            height,
            width,
            inpaint_padding,
            fix_steps,
            inpainting_mask_invert,

            images,
            generationsN,
            sd_model_checkpoint,
            mask_num,
            None,
            prepareMask(avoidance_mask_mode, avoidance_mask),
            only_custom_mask,
            prepareMask(custom_mask_mode, custom_mask),
            use_inpaint_diff and inpaint_diff_mask_view is not None and \
                replacer_scripts.InpaintDifferenceGlobals is not None and \
                replacer_scripts.InpaintDifferenceGlobals.generated_mask is not None,

            *replacer_scripts.prepareScriptsArgs(scripts_args),
            )

        i = 1
        n = generationsN
        processed = None
        allExtraImages = []
        batch_processed = None

        for image in images:
            if shared.state.interrupted:
                if needAutoUnloadModels():
                    clearCache()
                break
            
            progressInfo = "Generate mask"
            if n > 1: 
                print(flush=True)
                print()
                print(f'    [{EXT_NAME}]    processing {i}/{n}')
                progressInfo += f" {i}/{n}"

            shared.state.textinfo = progressInfo
            shared.state.skipped = False

            saveDir = ""
            save_to_dirs = True
            if tab_index == 2 and output_batch_dir != "":
                saveDir = output_batch_dir
                save_to_dirs = False
            elif tab_index == 3:
                saveDir = video_output_dir
                save_to_dirs = False
            else:
                saveDir = getSaveDir()

            try:
                processed, extraImages = generateSingle(image, gArgs, saveDir, "", save_to_dirs,
                    extra_includes, batch_processed)
            except Exception as e:
                print(f'    [{EXT_NAME}]    Exception: {e}')

                i += 1
                if needAutoUnloadModels():
                    clearCache()
                if generationsN == 1:
                    raise
                if tab_index == 3:
                    save_image(image, saveDir, "", gArgs.seed, gArgs.positvePrompt,
                            opts.samples_format, save_to_dirs=False)
                shared.state.nextjob()
                continue

            allExtraImages += extraImages
            batch_processed = processed
            i += 1

        if processed is None:
            return [], "", plaintext_to_html(f"No one image was processed. See console logs for exceptions"), ""

        if tab_index == 1:
            gArgs.images = getImages(image_batch)
        if tab_index == 2:
            gArgs.images = readImages(input_batch_dir)
        if tab_index == 3:
            shared.state.textinfo = 'video saving'
            print("generate done, generating video")
            save_video_path = os.path.join(video_output_dir, f'output_{os.path.basename(input_video)}_{seed}.mp4')
            if len(save_video_path) > 260:
                save_video_path = os.path.join(video_output_dir, f'output_{seed}.mp4')
            save_video(video_output_dir, fps_out, input_video, save_video_path, seed)


        global lastGenerationArgs
        lastGenerationArgs = gArgs
        lastGenerationArgs.hiresFixCacheData = HiresFixCacheData(gArgs.upscalerForImg2Img, processed.images[0])
        shared.state.end()

        if tab_index == 3:
            return [], "", plaintext_to_html(f"Saved into {save_video_path}"), ""
        
        if tab_index == 2 and not show_batch_dir_results:
            return [], "", plaintext_to_html(f"Saved into {output_batch_dir}"), ""

        processed.images += allExtraImages
        processed.infotexts += [processed.info] * len(allExtraImages)

        return processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")
    finally:
        for restore in restoreList:
            restore()





def applyHiresFix(
    hf_upscaler,
    hf_steps,
    hf_sampler,
    hf_denoise,
    hf_cfg_scale,
    hfPositivePromptSuffix,
    hf_size_limit,
    hf_above_limit_upscaler,
    hf_unload_detection_models,
    hf_disable_cn,
    hf_extra_mask_expand,
    hf_positvePrompt,
    hf_negativePrompt,
    hf_sd_model_checkpoint,
    hf_extra_inpaint_padding,
    hf_extra_mask_blur,
    hf_randomize_seed,
):
    if hfPositivePromptSuffix == "":
        hfPositivePromptSuffix = getHiresFixPositivePromptSuffixExamples()[0]


    global lastGenerationArgs
    if lastGenerationArgs is None:
        return [], "", plaintext_to_html("no last generation data"), ""

    gArgs = copy.copy(lastGenerationArgs)
    gArgs.upscalerForImg2Img = hf_upscaler

    hrArgs = copy.copy(lastGenerationArgs)
    hrArgs.cfg_scale = hf_cfg_scale
    hrArgs.denoising_strength = hf_denoise
    if not hf_sampler == 'Use same sampler':
        hrArgs.sampler_name = hf_sampler
    if hf_steps != 0:
        hrArgs.steps = hf_steps
    if hf_extra_mask_expand != 0:
        hrArgs.mask = extraMaskExpand(hrArgs.mask, hf_extra_mask_expand)
    hrArgs.inpainting_fill = 1 # Original
    hrArgs.img2img_fix_steps = True
    if hf_disable_cn:
        hrArgs.cn_args = None
    if hf_positvePrompt != "":
        hrArgs.positvePrompt = hf_positvePrompt
    hrArgs.positvePrompt = hrArgs.positvePrompt + " " + hfPositivePromptSuffix
    if hf_negativePrompt != "":
        hrArgs.negativePrompt = hf_negativePrompt
    if hf_sd_model_checkpoint is not None and hf_sd_model_checkpoint != 'Use same model'\
            and hf_sd_model_checkpoint != "":
        hrArgs.sd_model_checkpoint = hf_sd_model_checkpoint
    hrArgs.inpaint_full_res_padding += hf_extra_inpaint_padding
    hrArgs.mask_blur += hf_extra_mask_blur
    if hf_randomize_seed:
        hrArgs.seed = generateSeed()

    if gArgs.generationsN > 1 or gArgs.batch_size > 1 or gArgs.n_iter > 1:
        errorText = f"    [{EXT_NAME}]    applyHiresFix is not supported for batch"
        print(errorText)
        return None, "", plaintext_to_html(errorText), ""


    if hf_unload_detection_models:
        clearCache()

    shared.state.begin(job=f'{EXT_NAME_LOWER}_hf')
    shared.state.job_count = 2
    shared.total_tqdm.clear()
    shared.total_tqdm.updateTotal(gArgs.steps + hrArgs.steps)

    image = None
    for image_ in gArgs.images:
        image = image_
        break

    saveDir = getSaveDir()
    hrArgs.width, hrArgs.height = image.size
    if hrArgs.height > hf_size_limit:
        hrArgs.height = hf_size_limit
        hrArgs.upscalerForImg2Img = hf_above_limit_upscaler
    if hrArgs.width > hf_size_limit:
        hrArgs.width = hf_size_limit
        hrArgs.upscalerForImg2Img = hf_above_limit_upscaler

    shared.state.textinfo = "inpaint with upscaler"
    if lastGenerationArgs.hiresFixCacheData is not None and\
            lastGenerationArgs.hiresFixCacheData.upscaler == hf_upscaler:
        generatedImage = lastGenerationArgs.hiresFixCacheData.generatedImage
        print('hiresFixCacheData restored from cache')
        shared.state.job_count = 1
        shared.total_tqdm.updateTotal(hrArgs.steps)
    else:
        processed, scriptImages = inpaint(image, gArgs)
        generatedImage = processed.images[0]
        lastGenerationArgs.hiresFixCacheData = HiresFixCacheData(hf_upscaler, generatedImage)
        print('hiresFixCacheData cached')
        

    shared.state.textinfo = "hiresfix"
    processed, scriptImages = inpaint(generatedImage, hrArgs, saveDir, "-hires-fix")

    shared.state.end()

    return processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")




def generate_webui(id_task, *args, **kwargs):
    return generate(*args, **kwargs)

def applyHiresFix_webui(id_task, *args, **kwargs):
    return applyHiresFix(*args, **kwargs)
