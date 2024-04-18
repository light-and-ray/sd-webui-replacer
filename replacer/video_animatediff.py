import os, copy, math
from PIL import Image, ImageChops
from tqdm import tqdm
from modules import shared, errors
from replacer.generation_args import GenerationArgs
from replacer.mask_creator import createMask, NothingDetectedError
from replacer.inpaint import inpaint
from replacer.generate import generateSingle
from replacer.tools import ( interrupted, applyMaskBlur, clearCache, limitImageByOneDemention,
    Pause, extraMaskExpand,
)



def processFragment(fragmentPath: str, initImage, gArgs: GenerationArgs):
    gArgs = copy.copy(gArgs)
    gArgs.inpainting_mask_invert = False
    gArgs.animatediff_args = copy.copy(gArgs.animatediff_args)
    gArgs.animatediff_args.needApplyAnimateDiff = True
    gArgs.animatediff_args.video_path = os.path.join(fragmentPath, 'frames')
    gArgs.animatediff_args.mask_path = os.path.join(fragmentPath, 'masks')
    processed, _ = inpaint(initImage, gArgs)

    outDir = os.path.join(fragmentPath, 'out')
    for idx in range(len(processed.images)):
        processed.images[idx].save(os.path.join(outDir, f'frame_{idx}.png'))

    return processed



def getFragments(gArgs: GenerationArgs, video_output_dir: str, totalFragments: int):
    fragmentSize = gArgs.animatediff_args.fragment_length

    frames = gArgs.images
    blackFilling = Image.new('L', frames[0].size, 0).convert('RGBA')
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
                shared.state.textinfo = f"inpainting fragment {fragmentNum} / {totalFragments}"
                yield fragmentPath
            frameInFragmentIdx = 0
            fragmentNum += 1
            fragmentPath = os.path.join(video_output_dir, f"fragment_{fragmentNum}")
            
            framesDir = os.path.join(fragmentPath, 'frames'); os.makedirs(framesDir, exist_ok=True)
            masksDir = os.path.join(fragmentPath, 'masks'); os.makedirs(masksDir, exist_ok=True)
            outDir = os.path.join(fragmentPath, 'out'); os.makedirs(outDir, exist_ok=True)

            # last frame goes first in the next fragment
            if mask is not None:
                frame.save(os.path.join(framesDir, f'frame_{frameInFragmentIdx}.png'))
                mask.save(os.path.join(masksDir, f'frame_{frameInFragmentIdx}.png'))
                frameInFragmentIdx = 1

        Pause.wait()
        if interrupted(): return
        shared.state.textinfo = f"generating masks for fragment {fragmentNum} / {totalFragments}"
        print(f"    {frameInFragmentIdx+1} / {fragmentSize}")

        frame = frames[frameIdx]
        frame.save(os.path.join(framesDir, f'frame_{frameInFragmentIdx}.png'))
        try:
            mask = createMask(frame, gArgs).mask
            if gArgs.inpainting_mask_invert:
                mask = ImageChops.invert(mask.convert('L'))
            mask = applyMaskBlur(mask.convert('RGBA'), gArgs.mask_blur)
            mask = mask.resize(frame.size)
        except Exception as e:
            if type(e) is not NothingDetectedError:
                errors.report(f'{e} ***', exc_info=True)
            else:
                print(e)
            if mask is None or mask is blackFilling:
                mask = blackFilling
            else:
                mask = extraMaskExpand(mask, 50)
        mask.save(os.path.join(masksDir, f'frame_{frameInFragmentIdx}.png'))
        frameInFragmentIdx += 1
    if frameInFragmentIdx > 1:
        yield fragmentPath



def animatediffGenerate(gArgs: GenerationArgs, video_output_dir: str, result_dir, video_fps: float):
    if gArgs.animatediff_args.force_override_sd_model:
        gArgs.override_sd_model = True
        gArgs.sd_model_checkpoint = gArgs.animatediff_args.force_sd_model_checkpoint
    if gArgs.animatediff_args.internal_fps <= 0:
        gArgs.animatediff_args.internal_fps = video_fps
    if gArgs.animatediff_args.fragment_length <= 0 or len(gArgs.images) < gArgs.animatediff_args.fragment_length:
        gArgs.animatediff_args.fragment_length = len(gArgs.images)
    gArgs.animatediff_args.needApplyCNForAnimateDiff = True

    totalFragments = math.ceil((len(gArgs.images) - 1) / (gArgs.animatediff_args.fragment_length - 1))
    if gArgs.animatediff_args.generate_only_first_fragment:
        totalFragments = 1
    shared.state.job_count = 1 + totalFragments
    shared.total_tqdm.clear()
    Pause.paused = False

    shared.state.textinfo = f"processing the first frame. Total fragments number = {totalFragments}"
    processedFirstImg, _ = generateSingle(gArgs.images[0], copy.copy(gArgs), "", "", False, [], None)
    initImage: Image = processedFirstImg.images[0]
    fragmentPaths = []


    for fragmentPath in getFragments(gArgs, video_output_dir, totalFragments):
        if not shared.cmd_opts.lowram: # do not confuse with lowvram. lowram is for really crazy people
            clearCache()
        processed = processFragment(fragmentPath, initImage, gArgs)
        if interrupted():
            break
        fragmentPaths.append(fragmentPath)
        initImage = processed.images[-1]
        if gArgs.animatediff_args.generate_only_first_fragment:
            break
    
    text = "merging fragments"
    shared.state.textinfo = text
    print(text)
    def readImages(input_dir):
        image_list = shared.listfiles(input_dir)
        for filename in image_list:
            image = Image.open(filename).convert('RGBA')
            yield image
    def saveImage(image):
        if not image: return
        image.save(os.path.join(result_dir, f"{frameNum:05d}-{gArgs.seed}.{shared.opts.samples_format}"))
    os.makedirs(result_dir, exist_ok=True)
    theLastImage = None
    frameNum = 0

    for fragmentPath in tqdm(fragmentPaths):
        images = list(readImages(os.path.join(fragmentPath, 'out')))
        if len(images) <= 1:
            break
        if theLastImage:
            images[0] = Image.blend(images[0], theLastImage, 0.5)
        theLastImage = images[-1]
        images = images[:-1]
        
        for image in images:
            saveImage(image)
            frameNum += 1
    saveImage(theLastImage)

