import subprocess
import cv2
import os
import glob
import modules.shared as shared
from PIL import Image

def separate_video_into_frames(video_path, fps, temp_folder):
    assert video_path, 'video not selected'
    assert temp_folder, 'temp folder not specified'

    # Create the temporary folder if it doesn't exist
    os.makedirs(temp_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    fps_in = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps_out = fps_in
    else:
        fps_out = fps
    print(fps_in, fps_out)
    
    index_in = -1
    index_out = -1

    # Read frames from the video and save them as images
    frame_count = 0
    while True:
        success = video.grab()
        if not success: break
        index_in += 1

        out_due = int(index_in / fps_in * fps_out)
        # print(index_in, out_due, index_out)
        if out_due > index_out:
            success, frame = video.retrieve()
            if not success: break
            index_out += 1
            # Save the frame as an image in the temporary folder
            frame_path = os.path.join(temp_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1

    # Release the video file
    video.release()
    return fps_in, fps_out


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


def getVideoFrames(video_path, fps):
    assert video_path, 'video not selected'
    temp_folder = os.path.join(os.path.dirname(video_path), 'temp')
    if os.path.exists(temp_folder):
        for file in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, file))
    fps_in, fps_out = separate_video_into_frames(video_path, fps, temp_folder)
    return readImages(temp_folder), fps_in, fps_out


def save_video(frames_dir, frames_fps, org_video, output_path, target_fps, seed):
    # frames = glob.glob(os.path.join(frames_dir, f'*{seed}*'))
    # frames.sort()
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(frames_fps),
        '-i', os.path.join(frames_dir, '%5d-' + f'{seed}' + '.png'),
        '-r', str(frames_fps),
        '-i', org_video,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-vf', f'fps={target_fps}',
        '-shortest',
        '-y',
        output_path
    ]
    print(' '.join(str(v) for v in ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd)