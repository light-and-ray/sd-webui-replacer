from PIL import Image, ImageOps
from dataclasses import dataclass
from modules import devices
from replacer.extensions import replacer_extensions
from replacer.generation_args import GenerationArgs
from replacer.options import EXT_NAME, useCpuForDetection, useFastDilation
from replacer.tools import areImagesTheSame, limitImageByOneDimension, fastMaskDilate, applyRotationFix, removeRotationFix
sam_predict = None
update_mask = None
clear_cache = None

def initSamDependencies():
    global sam_predict, update_mask, clear_cache
    if not sam_predict or not update_mask or not clear_cache:
        import scripts.sam
        sam_predict = scripts.sam.sam_predict
        if useFastDilation():
            update_mask = fastMaskDilate
        else:
            update_mask = scripts.sam.update_mask
        clear_cache = scripts.sam.clear_cache
        if useCpuForDetection():
            scripts.sam.sam_device = 'cpu'
            print('Use CPU for SAM')


class NothingDetectedError(Exception):
    def __init__(self):
        super().__init__("Nothing has been detected")



masksCreatorCached = None


class MasksCreator:
    def __init__(self, detectionPrompt, avoidancePrompt, image, samModel, grdinoModel, boxThreshold,
            maskExpand, maxResolutionOnDetection, avoidance_mask, custom_mask, rotation_fix):
        self.detectionPrompt = detectionPrompt
        self.avoidancePrompt = avoidancePrompt
        self.image = image
        self.samModel = samModel
        self.grdinoModel = grdinoModel
        self.boxThreshold = boxThreshold
        self.maskExpand = maskExpand
        self.maxResolutionOnDetection = maxResolutionOnDetection
        self.avoidance_mask = avoidance_mask
        self.custom_mask = custom_mask
        self.rotation_fix = rotation_fix

        global masksCreatorCached

        if masksCreatorCached is not None and \
                self.detectionPrompt == masksCreatorCached.detectionPrompt and\
                self.avoidancePrompt == masksCreatorCached.avoidancePrompt and\
                self.samModel == masksCreatorCached.samModel and\
                self.grdinoModel == masksCreatorCached.grdinoModel and\
                self.boxThreshold == masksCreatorCached.boxThreshold and\
                self.maskExpand == masksCreatorCached.maskExpand and\
                self.maxResolutionOnDetection == masksCreatorCached.maxResolutionOnDetection and\
                self.rotation_fix == masksCreatorCached.rotation_fix and\
                areImagesTheSame(self.image, masksCreatorCached.image) and\
                areImagesTheSame(self.avoidance_mask, masksCreatorCached.avoidance_mask) and\
                areImagesTheSame(self.custom_mask, masksCreatorCached.custom_mask):
            self.previews = masksCreatorCached.previews
            self.masks = masksCreatorCached.masks
            self.cut = masksCreatorCached.cut
            self.boxes = masksCreatorCached.boxes
            print('MasksCreator restored from cache')
        else:
            restoreList = []
            try:
                if useCpuForDetection():
                    oldDevice = devices.device
                    def restore():
                        devices.device = oldDevice
                    restoreList.append(restore)
                    devices.device = 'cpu'
                    print('Use CPU for detection')
                self._createMasks()
                masksCreatorCached = self
                print('MasksCreator cached')
            finally:
                for restore in restoreList:
                    restore()


    def _createMasks(self):
        initSamDependencies()
        self.previews = []
        self.masks = []
        self.cut = []
        self.boxes = []

        imageResized = limitImageByOneDimension(self.image, self.maxResolutionOnDetection)
        imageResized = applyRotationFix(imageResized, self.rotation_fix)
        if self.avoidance_mask is None:
            customAvoidanceMaskResized = None
        else:
            customAvoidanceMaskResized = self.avoidance_mask.resize(imageResized.size)
            customAvoidanceMaskResized = applyRotationFix(customAvoidanceMaskResized, self.rotation_fix)
        masks, samLog = sam_predict(self.samModel, imageResized, [], [], True,
            self.grdinoModel, self.detectionPrompt, self.boxThreshold, False, [])
        print(samLog)
        if len(masks) == 0:
            if self.custom_mask is not None:
                print(f'[{EXT_NAME}] nothing has been detected by detection prompt, but there is custom mask')
                self.masks = [self.custom_mask]
                return
            else:
                raise NothingDetectedError()
        boxes = [masks[0], masks[1], masks[2]]
        masks = [masks[3], masks[4], masks[5]]

        self.boxes = boxes

        for mask in masks:
            if self.maskExpand >= 0:
                expanded = update_mask(mask, 0, self.maskExpand, imageResized)
            else:
                mask = ImageOps.invert(mask.convert('L'))
                expanded = update_mask(mask, 0, -self.maskExpand, imageResized)
                mask = ImageOps.invert(expanded[1])
                expanded = update_mask(mask, 0, 0, imageResized)

            self.previews.append(expanded[0])
            self.masks.append(expanded[1])
            self.cut.append(expanded[2])

        if self.avoidancePrompt != "":
            detectedAvoidanceMasks, samLog = sam_predict(self.samModel, imageResized, [], [], True,
                self.grdinoModel, self.avoidancePrompt, self.boxThreshold, False, [])
            print(samLog)
            if len(detectedAvoidanceMasks) == 0:
                print(f'[{EXT_NAME}] nothing has been detected by avoidance prompt')
                if customAvoidanceMaskResized:
                    avoidanceMasks = [customAvoidanceMaskResized, customAvoidanceMaskResized, customAvoidanceMaskResized]
                else:
                    avoidanceMasks = None
            else:
                avoidanceMasks = [detectedAvoidanceMasks[3], detectedAvoidanceMasks[4], detectedAvoidanceMasks[5]]
                if customAvoidanceMaskResized is not None:
                    for i in range(len(avoidanceMasks)):
                        avoidanceMasks[i] = avoidanceMasks[i].convert('L')
                        avoidanceMasks[i].paste(customAvoidanceMaskResized, customAvoidanceMaskResized)
        else:
            if customAvoidanceMaskResized:
                avoidanceMasks = [customAvoidanceMaskResized, customAvoidanceMaskResized, customAvoidanceMaskResized]
            else:
                avoidanceMasks = None

        if avoidanceMasks is not None:
            for i in range(len(self.masks)):
                maskTmp = ImageOps.invert(self.masks[i].convert('L'))
                whiteFilling = Image.new('L', maskTmp.size, 255)
                maskTmp.paste(whiteFilling, avoidanceMasks[i])
                self.masks[i] = ImageOps.invert(maskTmp)
                self.previews[i].paste(imageResized, avoidanceMasks[i])
                transparent = Image.new('RGBA', imageResized.size, (255, 0, 0, 0))
                self.cut[i].paste(transparent, avoidanceMasks[i])

        if self.custom_mask is not None:
            self.custom_mask = applyRotationFix(self.custom_mask, self.rotation_fix)
            for i in range(len(self.masks)):
                whiteFilling = Image.new('L', self.masks[i].size, 255)
                self.masks[i].paste(whiteFilling, self.custom_mask.resize(self.masks[i].size))

        for i in range(len(self.masks)):
            self.masks[i] = removeRotationFix(self.masks[i], self.rotation_fix)
        for i in range(len(self.previews)):
            self.previews[i] = removeRotationFix(self.previews[i], self.rotation_fix)
        for i in range(len(self.cut)):
            self.cut[i] = removeRotationFix(self.cut[i], self.rotation_fix)
        for i in range(len(self.boxes)):
            self.boxes[i] = removeRotationFix(self.boxes[i], self.rotation_fix)



@dataclass
class MaskResult:
    mask: Image.Image
    maskPreview: Image.Image
    maskCut: Image.Image
    maskBox: Image.Image


def createMask(image: Image.Image, gArgs: GenerationArgs) -> MaskResult:
    maskPreview = None
    maskCut = None
    maskBox = None

    if gArgs.do_not_use_mask:
        mask = Image.new('L', image.size, 255)
    elif gArgs.use_inpaint_diff:
        mask = replacer_extensions.inpaint_difference.Globals.generated_mask.convert('L')

    elif gArgs.only_custom_mask and gArgs.custom_mask is not None:
        mask = gArgs.custom_mask

    else:
        masksCreator = MasksCreator(gArgs.detectionPrompt, gArgs.avoidancePrompt, image, gArgs.samModel,
            gArgs.grdinoModel, gArgs.boxThreshold, gArgs.maskExpand, gArgs.maxResolutionOnDetection,
            gArgs.avoidance_mask, gArgs.custom_mask, gArgs.rotation_fix)

        if masksCreator.previews != []:
            if gArgs.mask_num == 'Random':
                maskNum = gArgs.seed % len(masksCreator.previews)
            else:
                maskNum = int(gArgs.mask_num) - 1
            mask = masksCreator.masks[maskNum]
            gArgs.mask_num_for_metadata = maskNum + 1

            maskPreview = masksCreator.previews[maskNum]
            maskCut = masksCreator.cut[maskNum]
            maskBox = masksCreator.boxes[maskNum]
        else:
            mask = gArgs.custom_mask

    return MaskResult(mask, maskPreview, maskCut, maskBox)
