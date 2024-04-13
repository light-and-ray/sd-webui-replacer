import os, copy
from PIL import Image
from modules import shared
from replacer.generation_args import GenerationArgs
from replacer.mask_creator import createMask, MaskResult
from replacer.inpaint import inpaint
from replacer.generate import generateSingle
from replacer.tools import interrupted, applyMaskBlur, clearCache
from replacer.options import needAutoUnloadModels



def processFragment(fragmentPath: str, initImage, gArgs: GenerationArgs):
    gArgs = copy.copy(gArgs)
    gArgs.animatediff_args = copy.copy(gArgs.animatediff_args)
    gArgs.animatediff_args.needApplyAnimateDiff = True
    gArgs.animatediff_args.video_path = os.path.join(fragmentPath, 'frames')
    gArgs.animatediff_args.mask_path = os.path.join(fragmentPath, 'masks')
    processed, _ = inpaint(initImage, gArgs)

    outDir = os.path.join(fragmentPath, 'out')
    for idx in range(len(processed.images)):
        processed.images[idx].save(os.path.join(outDir, f'frame_{idx}.png'))

    return processed



def getFragments(gArgs: GenerationArgs, video_output_dir: str):
    fragmentSize = 16

    frames = gArgs.images
    fragmentNum = 0
    frameInFragmentIdx = fragmentSize
    fragmentPath = None
    framesDir = None
    masksDir = None
    outDir = None
    frame = None
    mask = None

    for frameIdx in range(len(frames)):
        if frameInFragmentIdx == fragmentSize:
            if fragmentPath is not None:
                shared.state.textinfo = f"inpainting fragment {fragmentNum}"
                yield fragmentPath
            frameInFragmentIdx = 0
            fragmentNum += 1
            fragmentPath = os.path.join(video_output_dir, f"fragment_{fragmentNum}")
            shared.state.textinfo = f"generating masks for fragment {fragmentNum}"

            framesDir = os.path.join(fragmentPath, 'frames'); os.makedirs(framesDir, exist_ok=True)
            masksDir = os.path.join(fragmentPath, 'masks'); os.makedirs(masksDir, exist_ok=True)
            outDir = os.path.join(fragmentPath, 'out'); os.makedirs(outDir, exist_ok=True)

            # last frame goes first in the next fragment
            if mask is not None:
                frame.save(os.path.join(framesDir, f'frame_{frameInFragmentIdx}.png'))
                mask.save(os.path.join(masksDir, f'frame_{frameInFragmentIdx}.png'))
                frameInFragmentIdx = 1

        if interrupted(): return
        frame = frames[frameIdx]
        frame.save(os.path.join(framesDir, f'frame_{frameInFragmentIdx}.png'))
        maskResult: MaskResult = createMask(frame, gArgs)
        mask = maskResult.mask.resize(frame.size)
        mask = applyMaskBlur(mask, gArgs.mask_blur)
        mask.save(os.path.join(masksDir, f'frame_{frameInFragmentIdx}.png'))
        frameInFragmentIdx += 1


def animatediffGenerate(gArgs: GenerationArgs, video_output_dir: str):
    shared.state.textinfo = "processing init frame"
    processedFirstImg, _ = generateSingle(gArgs.images[0], copy.copy(gArgs), "", "", False, [], None)
    initImage: Image = processedFirstImg.images[0]

    for fragmentPath in getFragments(gArgs, video_output_dir):
        # if needAutoUnloadModels():
        clearCache()
        processed = processFragment(fragmentPath, initImage, gArgs)
        initImage = processed.images[-1]
        break
