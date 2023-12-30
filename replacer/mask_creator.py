from PIL import ImageChops
import modules.shared as shared


class NothingDetectedError(Exception):
    def __init__(self):
        super().__init__("Nothing has been detected")


def is_images_the_same(image_one, image_two):
    diff = ImageChops.difference(image_one.convert('RGB'), image_two.convert('RGB'))

    if diff.getbbox():
        return False
    else:
        return True


masksCreatorCached = None


class MasksCreator:
    def __init__(self, detectionPrompt, image, samModel, grdinoModel, boxThreshold, maskExpand):
        self.detectionPrompt = detectionPrompt
        self.image = image
        self.samModel = samModel
        self.grdinoModel = grdinoModel
        self.boxThreshold = boxThreshold
        self.maskExpand = maskExpand

        global masksCreatorCached

        if masksCreatorCached is not None and \
                self.detectionPrompt == masksCreatorCached.detectionPrompt and\
                self.samModel == masksCreatorCached.samModel and\
                self.grdinoModel == masksCreatorCached.grdinoModel and\
                self.boxThreshold == masksCreatorCached.boxThreshold and\
                self.maskExpand == masksCreatorCached.maskExpand and\
                is_images_the_same(self.image, masksCreatorCached.image):
            self.previews = masksCreatorCached.previews
            self.masksExpanded = masksCreatorCached.masksExpanded
            self.cutted = masksCreatorCached.cutted
            print('MasksCreator restored from cache')
        else:
            self._createMasks()
            masksCreatorCached = self
            print('MasksCreator cached')


    def _createMasks(self):
        from scripts.sam import sam_predict, update_mask

        masks, samLog = sam_predict(self.samModel, self.image, [], [], True,
            self.grdinoModel, self.detectionPrompt, self.boxThreshold, False, [])
        print(samLog)
        if len(masks) == 0:
            raise NothingDetectedError()

        masks = [masks[3], masks[4], masks[5]]

        self.previews = []
        self.masksExpanded = []
        self.cutted = []

        for mask in masks:
            if shared.state.interrupted or shared.state.skipped:
                break

            expanded = update_mask(mask, 0, self.maskExpand, self.image)
            self.previews.append(expanded[0])
            self.masksExpanded.append(expanded[1])
            self.cutted.append(expanded[2])
            