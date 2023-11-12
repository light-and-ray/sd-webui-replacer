from PIL import Image, PngImagePlugin
from PIL import ImageChops
import gradio as gr
from modules.img2img import img2img
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
import modules.shared as shared
from modules.shared import opts, state
from contextlib import closing
import modules.scripts
import numpy as np
import os
import copy
import importlib
from functools import lru_cache
import random
from modules import paths
from modules.ui import plaintext_to_html
from scripts.replacer_options import getDetectionPromptExamples, getPositivePromptExamples, getNegativePromptExamples
from scripts.replacer_options import useFirstPositivePromptFromExamples, useFirstNegativePromptFromExamples
from scripts.replacer_options import getHiresFixPositivePromptSuffixExamples
from scripts.replacer_mask_creator import MasksCreator
from scripts.replacer_generation_args import GenerationArgs
from scripts.replacer_options import EXT_NAME, getSaveDir
from modules.images import save_image


    


def inpaint(
    image : Image,
    gArgs : GenerationArgs,
    savePath : str = "",
    saveSuffix : str = "",
    save_to_dirs : bool = True
):
    override_settings = {}
    if (gArgs.upscalerForImg2Img is not None and gArgs.upscalerForImg2Img != ""):
        override_settings["upscaler_for_img2img"] = gArgs.upscalerForImg2Img

    if gArgs.img2img_fix_steps is not None and gArgs.img2img_fix_steps != "":
        override_settings["img2img_fix_steps"] = gArgs.img2img_fix_steps

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
        inpainting_fill=gArgs.inpainting_fill,
        resize_mode=0,
        denoising_strength=gArgs.denoising_strength,
        image_cfg_scale=1.5,
        inpaint_full_res=True,
        inpaint_full_res_padding=gArgs.inpaint_full_res_padding,
        inpainting_mask_invert=False,
        override_settings=override_settings,
        do_not_save_samples=True,
    )

    p.extra_generation_params["Mask blur"] = gArgs.mask_blur
    p.extra_generation_params["Detection prompt"] = gArgs.detectionPrompt
    is_batch = (gArgs.n_iter > 1 or gArgs.batch_size > 1)
    p.seed = gArgs.seed


    # if shared.cmd_opts.enable_console_prompts:
    #     print(f"\nimg2img: {gArgs.positvePrompt}", file=shared.progress_print_out)

    with closing(p):
        if is_batch:
            pass
            # assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

            # process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args, to_scale=selected_scale_tab == 1, scale_by=scale_by, use_png_info=img2img_batch_use_png_info, png_info_props=img2img_batch_png_info_props, png_info_dir=img2img_batch_png_info_dir)

            # processed = Processed(p, [], p.seed, "")
        else:
            processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    # if opts.samples_log_stdout:
    #     print(generation_info_js)

    if savePath != "":
        for imageToSave in processed.images:
            save_image(imageToSave, savePath, "", gArgs.seed, gArgs.positvePrompt, opts.samples_format,
                    info=processed.info, p=p, suffix=saveSuffix, save_to_dirs=save_to_dirs)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")



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
    save_to_dirs : bool
):
    masksCreator = MasksCreator(gArgs.detectionPrompt, image, gArgs.samModel,
                gArgs.grdinoModel, gArgs.boxThreshold, gArgs.maskExpand)

    maskNum = gArgs.seed % len(masksCreator.previews)

    maskPreview = masksCreator.previews[maskNum]
    gArgs.mask = masksCreator.masksExpanded[maskNum]
    maskCutted = masksCreator.cutted[maskNum]

    resultImages, generation_info_js, processed_info, processed_comments = \
        inpaint(image, gArgs, savePath, saveSuffix, save_to_dirs)

    return resultImages, generation_info_js, processed_info, processed_comments



def generate(
    detectionPrompt: str,
    positvePrompt: str,
    negativePrompt: str,
    tab_index,
    image_single,
    image_batch,
    input_batch_dir,
    output_batch_dir,
    show_batch_dir_results,
    upscalerForImg2Img,
    seed,
    # progress=gr.Progress(track_tqdm=True),
    
) -> Image.Image:
    if detectionPrompt == '':
        detectionPrompt = getDetectionPromptExamples()[0]

    if positvePrompt == '' and useFirstPositivePromptFromExamples():
        positvePrompt = getPositivePromptExamples()[0]

    if negativePrompt == '' and useFirstNegativePromptFromExamples():
        negativePrompt = getNegativePromptExamples()[0]

    if (seed == -1):
        seed = int(random.randrange(4294967294))


    images = []
    if tab_index == 0:
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
        images = getImages(image_batch)
        generationsN = len(image_batch)


    if tab_index == 2:
        def readImages(input_dir):
            assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
            assert input_dir, 'input directory not selected'

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                try:
                    image = Image.open(filename).convert('RGBA')
                except Exception:
                    continue
                yield image
        images = readImages(input_batch_dir)
        generationsN = len(shared.listfiles(input_batch_dir))


    samModel = 'sam_hq_vit_l.pth'
    grdinoModel = 'GroundingDINO_SwinT_OGC (694MB)'
    boxThreshold = 0.3
    maskExpand = 35
    
    steps = 20
    sampler_name = 'DPM++ 2M SDE Karras'
    mask_blur = 4
    inpainting_fill = 0 # Fill
    n_iter = 1
    batch_size = 1
    cfg_scale = 5.5
    denoising_strength = 1.0
    height = 512
    width = 512
    inpaint_full_res_padding = 20
    img2img_fix_steps = False

    gArgs = GenerationArgs(
        positvePrompt,
        negativePrompt,
        detectionPrompt,
        None,
        upscalerForImg2Img,
        seed,
        samModel,
        grdinoModel,
        boxThreshold,
        maskExpand,
        
        steps,
        sampler_name,
        mask_blur,
        inpainting_fill,
        n_iter,
        batch_size,
        cfg_scale,
        denoising_strength,
        height,
        width,
        inpaint_full_res_padding,
        img2img_fix_steps,

        images,
        generationsN,
        )

    resultImages = []
    generation_info_js = ""
    processed_info = ""
    processed_comments = ""
    i = 1
    n = generationsN

    for image in images:
        if n > 1: 
            print(flush=True)
            print()
            print(f'    [{EXT_NAME}]    processing {i}/{n}')

        saveDir = ""
        save_to_dirs = True
        if tab_index == 2:
            saveDir = output_batch_dir
            save_to_dirs = False
        else:
            saveDir = getSaveDir()

        try:
            newImages, generation_info_js, processed_info, processed_comments = \
                    generateSingle(image, gArgs, saveDir, "", save_to_dirs)
        except IndexError:
            print(f'    [{EXT_NAME}]    Exception IndexError - nothing found with the detection prompt')
            i += 1
            continue

        if not (tab_index == 2 and not show_batch_dir_results):
            resultImages += newImages

        i += 1


    if tab_index == 1:
        gArgs.images = getImages(image_batch)
    if tab_index == 2:
        gArgs.images = readImages(input_batch_dir)

    global lastGenerationArgs
    lastGenerationArgs = gArgs


    return resultImages, generation_info_js, processed_info, processed_comments





def applyHiresFixSingle(
    image : Image,
    gArgs : GenerationArgs,
    hrArgs : GenerationArgs,
    saveDir : str,
):
    generatedImages, _, _, _ = inpaint(image, gArgs)

    resultImages = []
    generation_info_js = ""
    processed_info = ""
    processed_comments = ""
    n = len(generatedImages)
    if n > 1: 
        print(f'    [{EXT_NAME}]    hiresfix batch count*size {n} for single image')

    for generatedImage in generatedImages:
        newImages, generation_info_js, processed_info, processed_comments = \
            inpaint(generatedImage, hrArgs, saveDir, "-hires-fix")
        resultImages += newImages

    return resultImages, generation_info_js, processed_info, processed_comments




def applyHiresFix(
    hf_upscaler,
    hf_steps,
    hf_sampler,
    hf_denoise,
    hf_cfg_scale,
    hfPositivePromptSuffix,
):
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
    hrArgs.sampler_name = hf_sampler
    hrArgs.steps = hf_steps
    hrArgs.positvePrompt = gArgs.positvePrompt + " " + hfPositivePromptSuffix
    hrArgs.inpainting_fill = 1 # Original
    hrArgs.img2img_fix_steps = True

    if gArgs.generationsN > 1:
        errorText = f"    [{EXT_NAME}]    applyHiresFix is not supported for batch"
        print(errorText)
        return None, "", errorText, ""

    resultImages = []
    generation_info_js = ""
    processed_info = ""
    processed_comments = ""


    for image in gArgs.images:
        saveDir = getSaveDir()
        hrArgs.height, hrArgs.width = image.size

        resultImages, generation_info_js, processed_info, processed_comments = \
            applyHiresFixSingle(image, gArgs, hrArgs, saveDir)


    return resultImages, generation_info_js, processed_info, processed_comments
