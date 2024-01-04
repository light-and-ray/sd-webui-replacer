from PIL import ImageChops, Image, ImageOps
from replacer.options import needAutoUnloadModels
sam_predict = None
update_mask = None
clear_cache = None

def initSamDependencies():
    global sam_predict, update_mask, clear_cache
    if not sam_predict or not update_mask or not clear_cache:
        from scripts.sam import sam_predict as sam_predict_
        from scripts.sam import update_mask as update_mask_
        from scripts.sam import clear_cache as clear_cache_
        sam_predict = sam_predict_
        update_mask = update_mask_
        clear_cache = clear_cache_


def limitSizeByOneDemention(image: Image, size):
    h, w = image.size
    if h > w:
        if h > size:
            w = size / h * w
            h = size
    else:
        if w > size:
            h = size / w * h
            w = size

    return image.resize((int(h), int(w)))



class NothingDetectedError(Exception):
    def __init__(self):
        super().__init__("Nothing has been detected")


def areImagesTheSame(image_one, image_two):
    diff = ImageChops.difference(image_one.convert('RGB'), image_two.convert('RGB'))

    if diff.getbbox():
        return False
    else:
        return True


masksCreatorCached = None


class MasksCreator:
    def __init__(self, detectionPrompt, avoidancePrompt, image, samModel, grdinoModel, boxThreshold,
            maskExpand, resolutionOnDetection):
        self.detectionPrompt = detectionPrompt
        self.avoidancePrompt = avoidancePrompt
        self.image = image
        self.samModel = samModel
        self.grdinoModel = grdinoModel
        self.boxThreshold = boxThreshold
        self.maskExpand = maskExpand
        self.resolutionOnDetection = resolutionOnDetection

        global masksCreatorCached

        if masksCreatorCached is not None and \
                self.detectionPrompt == masksCreatorCached.detectionPrompt and\
                self.avoidancePrompt == masksCreatorCached.avoidancePrompt and\
                self.samModel == masksCreatorCached.samModel and\
                self.grdinoModel == masksCreatorCached.grdinoModel and\
                self.boxThreshold == masksCreatorCached.boxThreshold and\
                self.maskExpand == masksCreatorCached.maskExpand and\
                self.resolutionOnDetection == masksCreatorCached.resolutionOnDetection and\
                areImagesTheSame(self.image, masksCreatorCached.image):
            self.previews = masksCreatorCached.previews
            self.masks = masksCreatorCached.masks
            self.cutted = masksCreatorCached.cutted
            self.boxes = masksCreatorCached.boxes
            print('MasksCreator restored from cache')
        else:
            self._createMasks()
            masksCreatorCached = self
            print('MasksCreator cached')


    def _createMasks(self):
        initSamDependencies()
        imageResized = limitSizeByOneDemention(self.image, self.resolutionOnDetection)
        masks, samLog = sam_predict(self.samModel, imageResized, [], [], True,
            self.grdinoModel, self.detectionPrompt, self.boxThreshold, False, [])
        print(samLog)
        if len(masks) == 0:
            raise NothingDetectedError()
        boxes = [masks[0], masks[1], masks[2]]
        masks = [masks[3], masks[4], masks[5]]

        self.previews = []
        self.masks = []
        self.cutted = []
        self.boxes = boxes

        for mask in masks:
            expanded = update_mask(mask, 0, self.maskExpand, imageResized)
            self.previews.append(expanded[0])
            self.masks.append(expanded[1])
            self.cutted.append(expanded[2])

        if self.avoidancePrompt != "":
            avoidanceMasks, samLog = sam_predict(self.samModel, imageResized, [], [], True,
                self.grdinoModel, self.avoidancePrompt, self.boxThreshold, False, [])
            print(samLog)
            if len(avoidanceMasks) == 0:
                print('nothing has been detected by avoidance prompt')
            else:
                avoidanceMasks = [avoidanceMasks[3], avoidanceMasks[4], avoidanceMasks[5]]
                for i in range(len(self.masks)):
                    maskTmp = ImageOps.invert(self.masks[i].convert('L'))
                    avoidanceMasks[i] = avoidanceMasks[i].convert('L')
                    maskTmp.paste(avoidanceMasks[i], avoidanceMasks[i])
                    self.masks[i] = ImageOps.invert(maskTmp)
                    self.previews[i].paste(imageResized, avoidanceMasks[i])
                    transparent = Image.new('RGBA', imageResized.size, (255, 0, 0, 0))
                    self.cutted[i].paste(transparent, avoidanceMasks[i])

        for i in range(len(self.masks)):
            self.masks[i] = self.masks[i].resize(self.image.size)

        if needAutoUnloadModels():
            clear_cache()
