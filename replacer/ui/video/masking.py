import os, shutil, math, datetime
import gradio as gr
from PIL import Image, ImageOps
from modules import shared

from replacer.video_tools import separate_video_into_frames
from replacer.tools import limitImageByOneDimension, makePreview, pil_to_base64_jpeg, prepareMask, applyMaskBlur
from replacer.ui.tools_ui import prepareExpectedUIBehavior
from replacer.options import getLimitMaskEditingResolution
from replacer.generation_args import GenerationArgs, DUMMY_HIRESFIX_ARGS
from .project import getOriginalVideoPath, getFrames, getMasks

from replacer.video_animatediff import detectVideoMasks


def prepareMasksDir(project_path: str, fps_out: int):
    if not project_path:
        raise gr.Error("No project selected")
    shared.state.textinfo = "preparing mask dir"
    framesDir = os.path.join(project_path, 'frames')
    if os.path.exists(framesDir):
        assert framesDir.endswith('frames')
        shutil.rmtree(framesDir)
    os.makedirs(framesDir, exist_ok=True)

    originalVideo = getOriginalVideoPath(project_path)
    separate_video_into_frames(originalVideo, fps_out, framesDir, 'png')

    masksDir = os.path.join(project_path, 'masks')
    if os.path.exists(masksDir):
        timestamp = int(datetime.datetime.now().timestamp())
        oldMasksDir = os.path.join(project_path, "old masks")
        rmDir = os.path.join(oldMasksDir, str(timestamp))
        os.makedirs(oldMasksDir, exist_ok=True)
        shutil.move(masksDir, rmDir)
    os.makedirs(masksDir, exist_ok=True)


def saveMask(project_path: str, mask: Image.Image, number: int):
    masksDir = os.path.join(project_path, 'masks')
    savePath = os.path.join(masksDir, f'frame_{number}.{shared.opts.samples_format}')
    mask.convert('RGB').save(savePath, subsampling=0, quality=93)




def getMasksPreview(project_path: str, page: int):
    if not project_path:
        raise gr.Error("No project selected")
    frames = list(getFrames(project_path))
    masks = list(getMasks(project_path))
    totalFrames = len(masks)

    start = page*10
    end = min(page*10+10, totalFrames)
    frames = frames[start: end]
    masks = masks[start: end]

    for i in range(len(frames)):
        frames[i] = limitImageByOneDimension(frames[i], getLimitMaskEditingResolution())
        masks[i] = masks[i].resize(frames[i].size)

    composited: list[Image.Image] = []
    for frame, mask in zip(frames, masks):
        composited.append(makePreview(frame, mask))

    for i in range(len(composited), 10):
        composited.append(None)

    composited = [pil_to_base64_jpeg(x) for x in composited]

    return page, f"**Page {page+1}/{math.ceil(totalFrames/10)}**", *composited





def generateEmptyMasks(task_id, project_path: str, fps_out: int, only_the_first_fragment: bool, fragment_length):
    prepareMasksDir(project_path, fps_out)
    frames = list(getFrames(project_path))
    maxNum = len(frames)
    if only_the_first_fragment and fragment_length != 0:
        maxNum = fragment_length
    for i in range(maxNum):
        blackFilling = Image.new('L', frames[0].size, 0)
        saveMask(project_path, blackFilling, i)

    return getMasksPreview(project_path, page=0)




def generateDetectedMasks(task_id, project_path: str, fps_out: int, only_the_first_fragment: bool, fragment_length,
        detectionPrompt,
        avoidancePrompt,
        seed,
        sam_model_name,
        dino_model_name,
        box_threshold,
        mask_expand,
        mask_blur,
        max_resolution_on_detection,
        inpainting_mask_invert,
        mask_num,
        avoidance_mask_mode,
        avoidance_mask,
        only_custom_mask,
        custom_mask_mode,
        custom_mask,
        do_not_use_mask,
    ):

    gArgs = GenerationArgs(
        positivePrompt="",
        negativePrompt="",
        detectionPrompt=detectionPrompt,
        avoidancePrompt=avoidancePrompt,
        upscalerForImg2Img="",
        seed=seed,
        samModel=sam_model_name,
        grdinoModel=dino_model_name,
        boxThreshold=box_threshold,
        maskExpand=mask_expand,
        maxResolutionOnDetection=max_resolution_on_detection,

        steps=0,
        sampler_name="",
        scheduler="",
        mask_blur=mask_blur,
        inpainting_fill=0,
        batch_count=0,
        batch_size=0,
        cfg_scale=0,
        denoising_strength=0,
        height=0,
        width=0,
        inpaint_full_res_padding=False,
        img2img_fix_steps=False,
        inpainting_mask_invert=inpainting_mask_invert,

        images=[],
        override_sd_model=False,
        sd_model_checkpoint="",
        mask_num=mask_num,
        avoidance_mask=prepareMask(avoidance_mask_mode, avoidance_mask),
        only_custom_mask=only_custom_mask,
        custom_mask=prepareMask(custom_mask_mode, custom_mask),
        use_inpaint_diff=False,
        clip_skip=0,
        pass_into_hires_fix_automatically=False,
        save_before_hires_fix=False,
        do_not_use_mask=do_not_use_mask,
        rotation_fix=None,
        variation_seed=0,
        variation_strength=0,
        integer_only_masked=False,
        forbid_too_small_crop_region=False,
        correct_aspect_ratio=False,

        hires_fix_args=DUMMY_HIRESFIX_ARGS,
        cn_args=None,
        soft_inpaint_args=None,
        )
    prepareExpectedUIBehavior(gArgs)


    prepareMasksDir(project_path, fps_out)
    frames = list(getFrames(project_path))
    maxNum = len(frames)
    if only_the_first_fragment and fragment_length != 0:
        maxNum = fragment_length
    masksDir = os.path.join(project_path, 'masks')

    detectVideoMasks(gArgs, frames, masksDir, maxNum)

    return [], "", "", ""





def reloadMasks(project_path: str, page: int):
    if not project_path:
        raise gr.Error("No project selected")
    masks = getMasks(project_path)
    if not masks:
        raise gr.Error("This project doesn't have masks")
    totalFrames = len(list(masks))
    totalPages = math.ceil(totalFrames/10) - 1

    if page > totalPages or page < 0:
        page = 0
    return getMasksPreview(project_path, page=page)


def goNextPage(project_path: str, page: int):
    if not project_path:
        raise gr.Error("No project selected")
    masks = getMasks(project_path)
    if not masks:
        raise gr.Error("This project doesn't have masks")
    totalFrames = len(list(masks))
    totalPages = math.ceil(totalFrames/10) - 1
    page = page + 1
    if page > totalPages:
        page = 0
    return getMasksPreview(project_path, page=page)


def goPrevPage(project_path: str, page: int):
    if not project_path:
        raise gr.Error("No project selected")
    masks = getMasks(project_path)
    if not masks:
        raise gr.Error("This project doesn't have masks")
    page = page - 1
    if page < 0:
        totalFrames = len(list(masks))
        totalPages = math.ceil(totalFrames/10) - 1
        page = totalPages
    return getMasksPreview(project_path, page=page)


def goToPage(project_path: str, page: int):
    if not project_path:
        raise gr.Error("No project selected")
    page = page-1
    masks = getMasks(project_path)
    if not masks:
        raise gr.Error("This project doesn't have masks")
    totalFrames = len(list(masks))
    totalPages = math.ceil(totalFrames/10) - 1
    if page < 0 or page > totalPages:
        raise gr.Error(f"Page {page+1} is out of range [1, {totalPages+1}]")
    return getMasksPreview(project_path, page=page)




def processMasks(action: str, project_path: str, page: int, mask_blur: int, masksNew: list[Image.Image]):
    if not project_path:
        raise gr.Error("No project selected")
    masksOld = getMasks(project_path)
    if not masksOld:
        raise gr.Error("This project doesn't have masks")
    masksOld = list(masksOld)
    firstMaskIdx = page*10
    for idx in range(len(masksNew)):
        maskNew = masksNew[idx]
        if not maskNew: continue
        maskNew = maskNew['mask'].convert('L')
        maskNew = applyMaskBlur(maskNew, mask_blur)
        maskOld = masksOld[firstMaskIdx+idx].convert('L')

        if action == 'add':
            whiteFilling = Image.new('L', maskOld.size, 255)
            editedMask = maskOld
            editedMask.paste(whiteFilling, maskNew.resize(maskOld.size))
        elif action == 'sub':
            maskTmp = ImageOps.invert(maskOld)
            whiteFilling = Image.new('L', maskTmp.size, 255)
            maskTmp.paste(whiteFilling, maskNew.resize(maskOld.size))
            editedMask = ImageOps.invert(maskTmp)
        saveMask(project_path, editedMask, firstMaskIdx+idx)
    return getMasksPreview(project_path, page=page)


def addMasks(project_path: str, page: int, mask_blur: int, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10):
    processMasks('add', project_path, page, mask_blur, [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])
    return tuple([gr.update()] * 12)

def subMasks(project_path: str, page: int, mask_blur: int, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10):
    processMasks('sub', project_path, page, mask_blur, [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])
    return tuple([gr.update()] * 12)

