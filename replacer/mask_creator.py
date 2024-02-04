from PIL import Image, ImageOps
from modules import devices
from replacer.options import needAutoUnloadModels, EXT_NAME, useCpuForDetection, useCpuForSam
from replacer.tools import areImagesTheSame, limitSizeByOneDemention
sam_predict = None
update_mask = None
clear_cache = None

def initSamDependencies():
    global sam_predict, update_mask, clear_cache
    if not sam_predict or not update_mask or not clear_cache:
        import scripts.sam
        sam_predict = scripts.sam.sam_predict
        update_mask = scripts.sam.update_mask
        clear_cache = scripts.sam.clear_cache
        if useCpuForDetection() or useCpuForSam():
            scripts.sam.sam_device = 'cpu'
            print(f'[{EXT_NAME}] Use CPU for SAM')


class NothingDetectedError(Exception):
    def __init__(self):
        super().__init__("Nothing has been detected")



masksCreatorCached = None


class MasksCreator:
    def __init__(self, detectionPrompt, avoidancePrompt, image, samModel, grdinoModel, boxThreshold,
            maskExpand, maxResolutionOnDetection, avoidance_mask, custom_mask):
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

        global masksCreatorCached

        if masksCreatorCached is not None and \
                self.detectionPrompt == masksCreatorCached.detectionPrompt and\
                self.avoidancePrompt == masksCreatorCached.avoidancePrompt and\
                self.samModel == masksCreatorCached.samModel and\
                self.grdinoModel == masksCreatorCached.grdinoModel and\
                self.boxThreshold == masksCreatorCached.boxThreshold and\
                self.maskExpand == masksCreatorCached.maskExpand and\
                self.maxResolutionOnDetection == masksCreatorCached.maxResolutionOnDetection and\
                areImagesTheSame(self.image, masksCreatorCached.image) and\
                areImagesTheSame(self.avoidance_mask, masksCreatorCached.avoidance_mask) and\
                areImagesTheSame(self.custom_mask, masksCreatorCached.custom_mask):
            self.previews = masksCreatorCached.previews
            self.masks = masksCreatorCached.masks
            self.cutted = masksCreatorCached.cutted
            self.boxes = masksCreatorCached.boxes
            print('MasksCreator restored from cache')
        else:
            restore = []
            try:
                if useCpuForDetection():
                    oldDevice = devices.device
                    def restore_():
                        devices.device = oldDevice
                    restore.append(restore_)
                    devices.device = 'cpu'
                    print(f'[{EXT_NAME}] Use CPU for detection')
                self._createMasks()
                masksCreatorCached = self
                print('MasksCreator cached')
            finally:
                for f in restore:
                    f()


    def _createMasks(self):
        initSamDependencies()
        self.previews = []
        self.masks = []
        self.cutted = []
        self.boxes = []

        imageResized = limitSizeByOneDemention(self.image, self.maxResolutionOnDetection)
        if self.avoidance_mask is None:
            customAvoidanceMaskResized = None
        else:
            customAvoidanceMaskResized = self.avoidance_mask.resize(imageResized.size)
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
            self.cutted.append(expanded[2])

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
                self.cutted[i].paste(transparent, avoidanceMasks[i])

        if self.custom_mask is not None:
            for i in range(len(self.masks)):
                whiteFilling = Image.new('L', self.masks[i].size, 255)
                self.masks[i].paste(whiteFilling, self.custom_mask.resize(self.masks[i].size))

        if needAutoUnloadModels():
            clear_cache()
