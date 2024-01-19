import os
import copy
import random
from contextlib import closing
from PIL import Image
import modules.shared as shared
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts
from modules.ui import plaintext_to_html
from modules.images import save_image
from replacer.mask_creator import MasksCreator
from replacer.generation_args import GenerationArgs
from replacer.video_tools import getVideoFrames, save_video
from replacer.options import ( getDetectionPromptExamples, getPositivePromptExamples,
    getNegativePromptExamples, useFirstPositivePromptFromExamples, useFirstNegativePromptFromExamples,
    getHiresFixPositivePromptSuffixExamples, EXT_NAME, EXT_NAME_LOWER, getSaveDir, needAutoUnloadModels
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
    samples_filename_pattern : str = "",
    batch_processed : Processed = None
):
    override_settings = {}
    if (gArgs.upscalerForImg2Img is not None and gArgs.upscalerForImg2Img != ""):
        override_settings["upscaler_for_img2img"] = gArgs.upscalerForImg2Img
    if gArgs.img2img_fix_steps is not None and gArgs.img2img_fix_steps != "":
        override_settings["img2img_fix_steps"] = gArgs.img2img_fix_steps
    if samples_filename_pattern != "":
        override_settings["samples_filename_pattern"] = samples_filename_pattern

    inpainting_fill = gArgs.inpainting_fill
    if (inpainting_fill == 4): # lama cleaner (https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content)
        inpainting_fill = 1 # original
        try:
            from lama_cleaner_masked_content.inpaint import lamaInpaint
            from lama_cleaner_masked_content.options import getUpscaler
            image = lamaInpaint(image, gArgs.mask, gArgs.inpainting_mask_invert, getUpscaler())
        except Exception as e:
            print(f'[{EXT_NAME}]: {e}')

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
        mask=gArgs.mask,
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
    p.extra_generation_params["Detection prompt"] = gArgs.detectionPrompt
    p.seed = gArgs.seed
    p.do_not_save_grid = not gArgs.save_grid
    


    with closing(p):
        processed = process_images(p)


    if savePath != "":
        for i in range(len(processed.images)):
            save_image(processed.images[i], savePath, "", processed.all_seeds[i], gArgs.positvePrompt, opts.samples_format,
                    info=processed.infotext(p, i), p=p, suffix=saveSuffix, save_to_dirs=save_to_dirs)

    if opts.do_not_show_images:
        processed.images = []

    if batch_processed:
        batch_processed.images += processed.images
        batch_processed.infotexts += processed.infotexts
        processed = batch_processed

    return processed



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
    samples_filename_pattern : str,
    batch_processed : list,
):
    masksCreator = MasksCreator(gArgs.detectionPrompt, gArgs.avoidancePrompt, image, gArgs.samModel,
        gArgs.grdinoModel, gArgs.boxThreshold, gArgs.maskExpand, gArgs.maxResolutionOnDetection)

    maskNum = gArgs.seed % len(masksCreator.previews)

    maskPreview = masksCreator.previews[maskNum]
    gArgs.mask = masksCreator.masks[maskNum]
    maskCutted = masksCreator.cutted[maskNum]
    maskBox = masksCreator.boxes[maskNum]
    shared.state.assign_current_image(maskPreview)
    shared.state.textinfo = "inpaint"

    processed = inpaint(image, gArgs, savePath, saveSuffix, save_to_dirs,
        samples_filename_pattern, batch_processed)

    extraImages = []
    if "mask" in extra_includes:
        extraImages.append(gArgs.mask)
    if "box" in extra_includes:
        extraImages.append(maskBox)
    if "cutted" in extra_includes:
        extraImages.append(maskCutted)
    if "preview" in extra_includes:
        extraImages.append(maskPreview)

    return processed, extraImages



def generate(
    detectionPrompt: str,
    avoidancePrompt: str,
    positvePrompt: str,
    negativePrompt: str,
    tab_index,
    image_single,
    image_batch,
    input_batch_dir,
    output_batch_dir,
    show_batch_dir_results,
    input_video,
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
    save_grid,
    extra_includes,
):
    shared.state.begin(job=EXT_NAME_LOWER)
    shared.total_tqdm.clear()

    if detectionPrompt == '':
        detectionPrompt = getDetectionPromptExamples()[0]
    detectionPrompt = detectionPrompt.strip()

    if positvePrompt == '' and useFirstPositivePromptFromExamples():
        positvePrompt = getPositivePromptExamples()[0]

    if negativePrompt == '' and useFirstNegativePromptFromExamples():
        negativePrompt = getNegativePromptExamples()[0]

    if (seed == -1):
        seed = int(random.randrange(4294967294))

    avoidancePrompt = avoidancePrompt.strip()

    output_batch_dir.strip()

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
                    image = Image.open(os.path.abspath(img.name)).convert('RGBA')
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
                except Exception:
                    continue
                yield image
        images = readImages(input_batch_dir)
        generationsN = len(shared.listfiles(input_batch_dir))


    if tab_index == 3:
        shared.state.textinfo = 'video preparing'
        temp_batch_folder = os.path.join(os.path.dirname(input_video), 'temp')
        if output_batch_dir == "":
            output_batch_dir = os.path.join(os.path.dirname(input_video), f'out_{seed}')
        else:
            output_batch_dir = os.path.join(output_batch_dir, f'out_{seed}')
        if os.path.exists(output_batch_dir):
            for file in os.listdir(output_batch_dir):
                if file.endswith(f'.{shared.opts.samples_format}'):
                    os.remove(os.path.join(output_batch_dir, file))
        images, fps_in, fps_out = getVideoFrames(input_video, target_video_fps)
        generationsN = len(shared.listfiles(temp_batch_folder))

        batch_count = 1
        batch_size = 1
        extra_includes = []
        save_grid = False

    if generationsN == 0:
        return [], "", plaintext_to_html(f"no input images"), ""
    shared.state.job_count = generationsN*batch_count

    img2img_fix_steps = False

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
        img2img_fix_steps,
        inpainting_mask_invert,

        images,
        generationsN,
        save_grid,
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
        if (tab_index == 2 or tab_index == 3) and output_batch_dir != "":
            saveDir = output_batch_dir
            save_to_dirs = False
        else:
            saveDir = getSaveDir()

        samples_filename_pattern = ""
        if tab_index == 3:
            samples_filename_pattern = "[seed]"

        try:
            processed, extraImages = generateSingle(image, gArgs, saveDir, "", save_to_dirs, extra_includes,
                    samples_filename_pattern, batch_processed)
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
        save_video_path = os.path.join(output_batch_dir, f'output_{os.path.basename(input_video)}_{seed}.mp4')
        save_video(output_batch_dir, fps_out, input_video, save_video_path, seed)


    global lastGenerationArgs
    lastGenerationArgs = gArgs
    shared.state.end()

    if tab_index == 3:
        return [], "", plaintext_to_html(f"Saved into {save_video_path}"), ""
    
    if tab_index == 2 and not show_batch_dir_results:
        return [], "", plaintext_to_html(f"Saved into {output_batch_dir}"), ""

    processed.images += allExtraImages
    processed.infotexts += [processed.info] * len(allExtraImages)

    return processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")





def applyHiresFixSingle(
    image : Image,
    gArgs : GenerationArgs,
    hrArgs : GenerationArgs,
    saveDir : str,
):
    shared.state.textinfo = "inpaint with upscaler"
    processed = inpaint(image, gArgs)
    generatedImages = processed.images

    n = len(generatedImages)
    if n > 1: 
        print(f'    [{EXT_NAME}]    hiresfix batch count*size {n} for single image')

    for generatedImage in generatedImages:
        shared.state.textinfo = "hiresfix"
        processed = inpaint(generatedImage, hrArgs, saveDir, "-hires-fix")

    return processed




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
):
    shared.state.begin(job=f'{EXT_NAME_LOWER}_hf')
    shared.state.job_count = 2
    shared.total_tqdm.clear()

    if hfPositivePromptSuffix == "":
        hfPositivePromptSuffix = getHiresFixPositivePromptSuffixExamples()[0]


    global lastGenerationArgs
    if lastGenerationArgs is None:
        return [], "", "", ""

    gArgs = copy.copy(lastGenerationArgs)
    gArgs.upscalerForImg2Img = hf_upscaler

    hrArgs = copy.copy(lastGenerationArgs)
    hrArgs.cfg_scale = hf_cfg_scale
    hrArgs.denoising_strength = hf_denoise
    if not hf_sampler == 'Use same sampler':
        hrArgs.sampler_name = hf_sampler
    if hf_steps != 0:
        hrArgs.steps = hf_steps
    hrArgs.positvePrompt = gArgs.positvePrompt + " " + hfPositivePromptSuffix
    hrArgs.inpainting_fill = 1 # Original
    hrArgs.img2img_fix_steps = True

    if gArgs.generationsN > 1 or gArgs.batch_size > 1 or gArgs.n_iter > 1:
        errorText = f"    [{EXT_NAME}]    applyHiresFix is not supported for batch"
        print(errorText)
        return None, "", errorText, ""


    if hf_unload_detection_models:
        clearCache()

    for image in gArgs.images:
        saveDir = getSaveDir()
        hrArgs.width, hrArgs.height = image.size
        if hrArgs.height > hf_size_limit:
            hrArgs.height = hf_size_limit
            hrArgs.upscalerForImg2Img = hf_above_limit_upscaler
        if hrArgs.width > hf_size_limit:
            hrArgs.width = hf_size_limit
            hrArgs.upscalerForImg2Img = hf_above_limit_upscaler

        processed : Processed = applyHiresFixSingle(image, gArgs, hrArgs, saveDir)

    shared.state.end()

    return processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")


def generate_webui(id_task, *args, **kwargs):
    return generate(*args, **kwargs)

def applyHiresFix_webui(id_task, *args, **kwargs):
    return applyHiresFix(*args, **kwargs)
