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
from scripts.replacer_mask_creator import MasksCreator





def inpaint(
    positvePrompt,
    negativePrompt,
    detectionPrompt,
    image,
    mask,
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
    seed,
):

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=positvePrompt,
        negative_prompt=negativePrompt,
        styles=[],
        sampler_name=sampler_name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=0,
        denoising_strength=denoising_strength,
        image_cfg_scale=1.5,
        inpaint_full_res=True,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=False,
        override_settings=[],
        do_not_save_samples=True,
    )

    p.extra_generation_params["Mask blur"] = mask_blur
    p.extra_generation_params["Detection prompt"] = detectionPrompt
    is_batch = (n_iter > 1 or batch_size > 1)
    p.seed = seed


    if shared.cmd_opts.enable_console_prompts:
        print(f"\nimg2img: {positvePrompt}", file=shared.progress_print_out)

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
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")



def generate(
    detectionPrompt: str,
    positvePrompt: str,
    negativePrompt: str,
    tab_index,
    image,
    image_batch,
    input_batch_dir,
    output_batch_dir,
    show_batch_dir_results,
    # progress=gr.Progress(track_tqdm=True),
    
) -> Image.Image:
    if detectionPrompt == '':
        detectionPrompt = getDetectionPromptExamples()[0]

    if positvePrompt == '' and useFirstPositivePromptFromExamples():
        positvePrompt = getPositivePromptExamples()[0]

    if negativePrompt == '' and useFirstNegativePromptFromExamples():
        negativePrompt = getNegativePromptExamples()[0]
    
    samModel = 'sam_hq_vit_l.pth'
    grdinoModel = 'GroundingDINO_SwinT_OGC (694MB)'
    boxThreshold = 0.3

    masksCreator = MasksCreator(detectionPrompt, image, samModel, grdinoModel, boxThreshold)

    seed = int(random.randrange(4294967294))
    maskNum = seed % len(masksCreator.previews)

    maskPreview = masksCreator.previews[maskNum]
    mask = masksCreator.masksExpanded[maskNum]
    maskCutted = masksCreator.cutted[maskNum]

    steps = 20
    sampler_name = 'DPM++ 2M SDE Karras'
    mask_blur = 4
    inpainting_fill = 0
    n_iter = 1
    batch_size = 1
    cfg_scale = 5.5
    denoising_strength = 1.0
    height = 512
    width = 512
    inpaint_full_res_padding = 20


    return inpaint(positvePrompt, negativePrompt, detectionPrompt, image, mask,
            steps, sampler_name, mask_blur, inpainting_fill, n_iter,
            batch_size, cfg_scale, denoising_strength,
            height, width, inpaint_full_res_padding, seed)

