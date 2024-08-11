import cv2, random, git, torch, os, time, urllib.parse, copy
import numpy as np
from PIL import ImageChops, Image, ImageColor
from dataclasses import dataclass
import gradio as gr
from modules.images import resize_image
from modules import errors, shared, masking
from modules.ui import versions_html
from replacer.generation_args import GenerationArgs
from replacer.options import useFastDilation, getMaskColorStr, EXT_ROOT_DIRECTORY, EXT_NAME

try:
    REPLACER_VERSION = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
except Exception:
    errors.report(f"Error reading replacer git info from {__file__}", exc_info=True)
    REPLACER_VERSION = "None"



def addReplacerMetadata(p, gArgs: GenerationArgs):
    p.extra_generation_params["Extension"] = f'sd-webui-replacer {REPLACER_VERSION}'
    if gArgs.detectionPrompt != '':
        p.extra_generation_params["Detection prompt"] = gArgs.detectionPrompt
    if gArgs.avoidancePrompt != '':
        p.extra_generation_params["Avoidance prompt"] = gArgs.avoidancePrompt
    p.extra_generation_params["Sam model"] = gArgs.samModel
    p.extra_generation_params["GrDino model"] = gArgs.grdinoModel
    p.extra_generation_params["Box threshold"] = gArgs.boxThreshold
    p.extra_generation_params["Mask expand"] = gArgs.maskExpand
    p.extra_generation_params["Max resolution on detection"] = gArgs.maxResolutionOnDetection
    if gArgs.mask_num_for_metadata is not None:
        p.extra_generation_params["Mask num"] = gArgs.mask_num_for_metadata
    if gArgs.addHiresFixIntoMetadata:
        pass


def areImagesTheSame(image_one, image_two):
    if image_one is None or image_two is None:
        return image_one is None and image_two is None
    if image_one.size != image_two.size:
        return False

    diff = ImageChops.difference(image_one.convert('RGB'), image_two.convert('RGB'))

    if diff.getbbox():
        return False
    else:
        return True


def limitSizeByOneDimension(size: tuple, limit: int) -> tuple:
    w, h = size
    if h > w:
        if h > limit:
            w = limit / h * w
            h = limit
    else:
        if w > limit:
            h = limit / w * h
            w = limit

    return (int(w), int(h))


def limitImageByOneDimension(image: Image.Image, limit: int) -> Image.Image:
    if image is None:
        return None
    return image.resize(limitSizeByOneDimension(image.size, limit))


def fastMaskDilate_(mask, dilation_amount):
    if dilation_amount == 0:
        return mask

    oldMode = mask.mode
    mask = np.array(mask.convert('RGB')).astype(np.int32)
    tensor_mask = torch.from_numpy((mask / 255).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_mask = tensor_mask.to(device)
    kernel = torch.ones(1, 1, 3, 3).to(device)

    tensor_mask_r = tensor_mask[:, 0:1, :, :]
    tensor_mask_g = tensor_mask[:, 1:2, :, :]
    tensor_mask_b = tensor_mask[:, 2:3, :, :]
    for _ in range(dilation_amount):
        tensor_mask_r = (torch.nn.functional.conv2d(tensor_mask_r, kernel, padding=1) > 0).float()
        tensor_mask_g = (torch.nn.functional.conv2d(tensor_mask_g, kernel, padding=1) > 0).float()
        tensor_mask_b = (torch.nn.functional.conv2d(tensor_mask_b, kernel, padding=1) > 0).float()

    tensor_mask = torch.cat((tensor_mask_r, tensor_mask_g, tensor_mask_b), dim=1)
    dilated_mask = tensor_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    dilated_mask = (dilated_mask * 255).astype(np.uint8)
    return Image.fromarray(dilated_mask).convert(oldMode)


def fastMaskDilate(mask, _, dilation_amount, imageResized):
    print("Dilation Amount: ", dilation_amount)
    dilated_mask = fastMaskDilate_(mask, dilation_amount // 2)
    maskFilling = Image.new('RGBA', dilated_mask.size, (0, 0, 0, 0))
    maskFilling.paste(Image.new('RGBA', dilated_mask.size, ImageColor.getcolor(f'{getMaskColorStr()}7F', 'RGBA')), dilated_mask)
    preview = imageResized.resize(dilated_mask.size)
    preview.paste(maskFilling, (0, 0), maskFilling)
    cutted = Image.new('RGBA', dilated_mask.size, (0, 0, 0, 0))
    cutted.paste(imageResized, dilated_mask)

    return [preview, dilated_mask, cutted]


@dataclass
class CachedExtraMaskExpand:
    mask: Image.Image
    expand: int
    result: Image.Image

cachedExtraMaskExpand: CachedExtraMaskExpand = None
update_mask = None


def extraMaskExpand(mask: Image.Image, expand: int):
    global cachedExtraMaskExpand, update_mask

    if cachedExtraMaskExpand is not None and\
            cachedExtraMaskExpand.expand == expand and\
            areImagesTheSame(cachedExtraMaskExpand.mask, mask):
        print('extraMaskExpand restored from cache')
        return cachedExtraMaskExpand.result
    else:
        if update_mask is None:
            if useFastDilation():
                update_mask = fastMaskDilate
            else:
                from scripts.sam import update_mask as update_mask_
                update_mask = update_mask_
        expandedMask = update_mask(mask, 0, expand, mask.convert('RGBA'))[1]
        cachedExtraMaskExpand = CachedExtraMaskExpand(mask, expand, expandedMask)
        print('extraMaskExpand cached')
        return expandedMask


def prepareMask(mask_mode, mask_raw):
    if mask_mode is None or mask_raw is None:
        return None
    mask = None
    if 'Upload mask' in mask_mode:
        mask = mask_raw['image'].convert('L')
    if 'Draw mask' in mask_mode:
        mask = Image.new('L', mask_raw['mask'].size, 0) if mask is None else mask
        draw_mask = mask_raw['mask'].convert('L')
        mask.paste(draw_mask, draw_mask)
        blackFilling = Image.new('L', mask.size, 0)
        if areImagesTheSame(blackFilling, mask):
            return None
    return mask


def applyMaskBlur(image_mask, mask_blur):
    originalMode = image_mask.mode
    if mask_blur > 0:
        np_mask = np.array(image_mask).astype(np.uint8)
        kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)
        image_mask = Image.fromarray(np_mask).convert(originalMode)
    return image_mask


def applyMask(res, orig, mask, gArgs):
    upscaler = gArgs.upscalerForImg2Img
    if upscaler == "":
        upscaler = None

    w, h = orig.size
    imageProc = resize_image(1, res.convert('RGB'), w, h, upscaler).convert('RGBA') # 1 - resize and crop
    mask = mask.convert('L')
    if gArgs.inpainting_mask_invert:
        mask = ImageChops.invert(mask)
    mask = mask.resize(orig.size)
    new = copy.copy(orig)
    new.paste(imageProc, mask)
    return new


def generateSeed():
    return int(random.randrange(4294967294))




def getReplacerFooter():
    footer = ""
    try:
        with open(os.path.join(EXT_ROOT_DIRECTORY, 'html', 'replacer_footer.html'), encoding="utf8") as file:
            footer = file.read()
        footer = footer.format(versions=versions_html()
            .replace('checkpoint: <a id="sd_checkpoint_hash">N/A</a>',
                f'replacer: <a href="https://github.com/light-and-ray/sd-webui-replacer/commit/{REPLACER_VERSION}">{REPLACER_VERSION}</a>'))
    except Exception as e:
        errors.report(f"Error getReplacerFooter: {e}", exc_info=True)
        return ""
    return footer


def interrupted():
    return shared.state.interrupted or getattr(shared.state, 'stopping_generation', False)


g_clear_cache = None
def clearCache():
    global g_clear_cache
    if g_clear_cache is None:
        from scripts.sam import clear_cache
        g_clear_cache = clear_cache
    g_clear_cache()


class Pause:
    paused = False

    @staticmethod
    def toggle():
        if shared.state.job == '':
            return
        Pause.paused = not Pause.paused
        text = f"    [{EXT_NAME}]: "
        text += "Paused" if Pause.paused else "Resumed"
        text += " batch generation"
        gr.Info(text)
        print(text)
        if Pause.paused:
            shared.state.textinfo = "will be paused"

    @staticmethod
    def wait():
        if not Pause.paused:
            return

        print(f"    [{EXT_NAME}] paused")
        while Pause.paused:
            shared.state.textinfo = "paused"
            time.sleep(0.2)
            if interrupted():
                return

        print(f"    [{EXT_NAME}] resumed")
        shared.state.textinfo = "resumed"


def convertIntoPath(string: str) -> str:
    string = string.strip()
    if not string: return string
    if len(string) > 3 and string[0] == string[-1] and string[0] in ('"', "'"):
        string = string[1:-1]

    schemes = ['file', 'fish']
    prefixes = [f'{x}://' for x in schemes]
    isURL = any(string.startswith(x) for x in prefixes)

    if not isURL:
        return string
    else:
        for prefix in prefixes:
            if string.startswith(prefix):
                string = urllib.parse.unquote(string.removeprefix(prefix))
                string = string.removeprefix(string.split('/')[0]) # removes user:password@host:port if exists
                return string

        errors.report("Can't be here")
        return string


def applyRotationFix(image: Image.Image, fix: str) -> Image.Image:
    if image is None:
        return None
    if fix == '-':
        return image
    if fix == '⟲':
        return image.transpose(Image.ROTATE_90)
    if fix == '🗘':
        return image.transpose(Image.ROTATE_180)
    if fix == '⟳':
        return image.transpose(Image.ROTATE_270)

def removeRotationFix(image: Image.Image, fix: str) -> Image.Image:
    if image is None:
        return None
    if fix == '-':
        return image
    if fix == '⟲':
        return image.transpose(Image.ROTATE_270)
    if fix == '🗘':
        return image.transpose(Image.ROTATE_180)
    if fix == '⟳':
        return image.transpose(Image.ROTATE_90)


def getActualCropRegion(mask: Image.Image, padding: int, forbid_too_small_crop_region: bool, integer_only_masked: bool):
    if hasattr(masking, 'get_crop_region_v2'):
        crop_region = masking.get_crop_region_v2(mask, padding)
    else:
        crop_region = masking.get_crop_region(mask, padding)

    if crop_region:
        x1, y1, x2, y2 = crop_region
        w = (x2-x1)
        h = (y2-y1)
        crop_region = masking.expand_crop_region(crop_region, w, h, mask.width, mask.height)
        if forbid_too_small_crop_region and hasattr(masking, 'expand_too_small_crop_region'):
            crop_region = masking.expand_too_small_crop_region(crop_region, w, h, mask.width, mask.height)
        if integer_only_masked and hasattr(masking, 'fix_crop_region_integer_scale'):
            crop_region = masking.fix_crop_region_integer_scale(crop_region, w, h, mask.width, mask.height)

    return crop_region

