import os, shutil, math
import gradio as gr
from PIL import Image
from modules import shared

from replacer.video_tools import separate_video_into_frames
from replacer.tools import limitImageByOneDimension, makePreview
from .project import getOriginalVideoPath, getFrames, getMasks


def prepareMasksDir(project_path: str, fps_out: int):
    if not project_path:
        raise gr.Error("No project selected")

    framesDir = os.path.join(project_path, 'frames')
    if os.path.exists(framesDir):
        assert framesDir.endswith('frames')
        shutil.rmtree(framesDir)
    os.makedirs(framesDir, exist_ok=True)

    originalVideo = getOriginalVideoPath(project_path)
    separate_video_into_frames(originalVideo, fps_out, framesDir)

    masksDir = os.path.join(project_path, 'masks')
    if os.path.exists(masksDir):
        assert masksDir.endswith('masks')
        shutil.rmtree(masksDir)
    os.makedirs(masksDir, exist_ok=True)


def saveMask(project_path: str, mask: Image.Image, number: int):
    masksDir = os.path.join(project_path, 'masks')
    savePath = os.path.join(masksDir, f'frame_{number}.{shared.opts.samples_format}')
    mask.convert('RGB').save(savePath, subsampling=0, quality=93)




def getMasksPreview(project_path: str, page: int):
    frames = list(getFrames(project_path))
    masks = list(getMasks(project_path))
    totalFrames = len(masks)

    start = page*10
    end = min(page*10+10, totalFrames)
    frames = frames[start: end]
    masks = masks[start: end]

    for i in range(len(frames)):
        frames[i] = limitImageByOneDimension(frames[i], 800)
        masks[i] = masks[i].resize(frames[i].size)

    composited: list[Image.Image] = []
    for frame, mask in zip(frames, masks):
        composited.append(makePreview(frame, mask))

    for i in range(len(composited), 10):
        composited.append(None)
    # for i in range(len(composited)):
    #     composited[i] = gr.update(value={'image': composited[i], 'mask': None})
    return page, f"**Page {page+1}/{math.ceil(totalFrames/10)}**", *composited




def generateEmptyMasks(project_path: str, fps_out: int, only_the_first_fragment: bool, fragment_length):
    prepareMasksDir(project_path, fps_out)
    frames = list(getFrames(project_path))
    maxNum = len(frames)
    if only_the_first_fragment and fragment_length != 0:
        maxNum = fragment_length
    for i in range(maxNum):
        blackFilling = Image.new('L', frames[0].size, 0)
        saveMask(project_path, blackFilling, i)

    return getMasksPreview(project_path, page=0)


def reloadMasks(project_path: str, page: int):
    masks = getMasks(project_path)
    if not masks:
        raise gr.Error("This project doesn't have masks")
    totalFrames = len(list(masks))
    totalPages = math.ceil(totalFrames/10)

    if page > totalPages:
        page = 0
    return getMasksPreview(project_path, page=page)


def goNextPage(project_path: str, page: int):
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
    page = page-1
    masks = getMasks(project_path)
    if not masks:
        raise gr.Error("This project doesn't have masks")
    totalFrames = len(list(masks))
    totalPages = math.ceil(totalFrames/10) - 1
    if page < 0 or page > totalPages:
        raise gr.Error(f"Page {page+1} is out of range 1, {totalPages+1}")
    return getMasksPreview(project_path, page=page)

