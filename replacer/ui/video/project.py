import os, glob, shutil, datetime
import gradio as gr
from PIL import Image
from replacer.tools import convertIntoPath, EXT_NAME
from replacer.video_tools import readImages


def getOriginalVideoPath(project_path: str):
    files = glob.glob(os.path.join(project_path, "original.*"))
    for file in files:
        if os.path.isfile(file) or os.path.islink(file):
            return file
    return None


def select(project_path: str):
    project_path = convertIntoPath(project_path)
    if not getOriginalVideoPath(project_path):
        return "❌ Selected path doesn't have original video", ""

    return f"✅ Selected a project {project_path!r}", project_path


def init(project_path: str, init_video: str):
    project_path = convertIntoPath(project_path)
    init_video = convertIntoPath(init_video)
    if not project_path:
        return "❌ Project path is not entered", "", gr.update()
    if not(os.path.isfile(init_video) or os.path.islink(init_video)):
        return "❌ Init video is not a file", "", gr.update()
    ext = os.path.basename(init_video).split('.')[-1]
    original_video = os.path.join(project_path, f'original.{ext}')
    os.makedirs(project_path, exist_ok=True)
    shutil.copy(init_video, original_video)
    return f"✅ Selected a new project {project_path!r}", project_path, project_path


def genNewProjectPath(init_video: str) -> str:
    init_video = convertIntoPath(init_video)
    if not init_video:
        return ""
    timestamp = int(datetime.datetime.now().timestamp())
    name = f'{EXT_NAME} project - {timestamp}'
    return os.path.join(os.path.dirname(init_video), name)


def getFrames(project_path: str):
    framesDir = os.path.join(project_path, 'frames')
    if not os.path.exists(framesDir):
        return None
    return readImages(framesDir)


def getMasks(project_path: str):
    framesDir = os.path.join(project_path, 'masks')
    if not os.path.exists(framesDir):
        return None
    return readImages(framesDir)
